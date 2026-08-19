import time
from typing import Optional

import torch

from src.common.tensor_op import pairwise_distance_matrix, create_class_matrix
from .metrics import calculate_metrics, aggregate_metrics, _metrics
from .reduction import reduce_chunk_distances

from src.common.utils import sec_to_hms


@torch.inference_mode()
def retrieve_and_evaluate(
    embeddings_q: torch.Tensor,
    clique_ids_q: torch.Tensor,
    version_ids_q: torch.Tensor,
    sizes_q: list[int],
    embeddings_db: torch.Tensor,
    clique_ids_db: torch.Tensor,
    version_ids_db: torch.Tensor,
    sizes_db: list[int],
    similarity_search: str = "MIPS",
    batch_size: int = 1,
    chunk_size: Optional[int] = None,
    k: Optional[int] = None,
) -> dict:
    """Perform similarity search for a set of embeddings and measure retrieval performance
    using the ground truth labels.

    Parameters:
    -----------
    embeddings: torch.Tensor
        2D tensor of shape (m, d) where m is the number of samples, d is the dimension
        of the embeddings. A sample may correspond to an entire track or a chunk of a track.
    labels: torch.Tensor
        1D tensor of shape (N,) where N is the number of tracks and labels[i] is the
        integer label of the i-th track.
    sizes: list
        List of length N where sizes[i] is the number of chunks of the i-th track.
        If granularity is "track", all elements are 1.
    similarity_search: str = "MIPS"
        The similarity search function to use. "NNS", "MCSS", or "MIPS".
    batch_size: int
        Number of tracks to process in each batch, to parallelize computations.
    chunk_size: int, optional
        Number of database tracks per chunk when computing the pairwise distance matrix.
        If None, each query batch is compared against the entire database at once.
        Smaller values trade speed for memory by capping the size of the intermediate
        (M, N_c_chunk) distance matrix.
    k: int, optional
        Number of similar tracks to consider for each query. If None, all tracks except
        the query itself are considered.

    Returns:
    --------
    metrics: dict
        Dictionary containing the performance metrics.

    """

    t0 = time.monotonic()
    print("Performing retrieval and calculating performance metrics...")

    assert 1 <= batch_size <= len(clique_ids_q)

    similarity_search = similarity_search.upper()
    if similarity_search not in ["NNS", "MCSS", "MIPS"]:
        raise ValueError

    assert clique_ids_q.ndim == 1, f"{clique_ids_q.shape}"
    assert version_ids_q.ndim == 1, f"{version_ids_q.shape}"
    assert embeddings_q.ndim == 2, f"{embeddings_q.shape}"
    assert (
        len(sizes_q) == clique_ids_q.shape[0] == version_ids_q.shape[0]
    ), f"{len(sizes_q)}, {clique_ids_q.shape[0]}, {version_ids_q.shape[0]}"

    assert clique_ids_db.ndim == 1, f"{clique_ids_db.shape}"
    assert version_ids_db.ndim == 1, f"{version_ids_db.shape}"
    assert embeddings_db.ndim == 2, f"{embeddings_db.shape}"
    assert (
        len(sizes_db) == clique_ids_db.shape[0] == version_ids_db.shape[0]
    ), f"{len(sizes_db)}, {clique_ids_db.shape[0]}, {version_ids_db.shape[0]}"

    # Segment and tracks stats
    N_c_db = sum(sizes_db)
    assert embeddings_db.shape[0] == N_c_db, "Embeddings and sizes do not match"
    N_t_db = len(sizes_db)
    print(f"Number of tracks in the database: {N_t_db:,}")
    print(f"Number of segments in the database: {N_c_db:,}")
    print(f"Maximum amount of segments a database track has: {max(sizes_db)}")

    N_c_q = sum(sizes_q)
    assert embeddings_q.shape[0] == N_c_q, "Embeddings and sizes do not match"
    N_t_q = len(sizes_q)
    print(f"Number of tracks in the queries: {N_t_q:,}")
    print(f"Number of segments in the queries: {N_c_q:,}")
    print(f"Maximum amount of segments a query track has: {max(sizes_q)}")

    if k is None:
        # Evaluate all results but exclude the query itself
        k = N_t_db - 1
    else:
        assert k <= N_t_db - 1

    # Precompute database chunk slices: (track_start, track_end, embed_start, embed_end)
    if chunk_size is None:
        db_chunks = [(0, N_t_db, 0, N_c_db)]
    else:
        assert chunk_size >= 1
        cum_sizes_db = torch.tensor([0] + sizes_db, dtype=torch.int).cumsum(0).tolist()
        db_chunks = [
            (
                ts,
                min(ts + chunk_size, N_t_db),
                cum_sizes_db[ts],
                cum_sizes_db[min(ts + chunk_size, N_t_db)],
            )
            for ts in range(0, N_t_db, chunk_size)
        ]

    # Track boundaries in the embeddings_q tensor
    boundaries_q = torch.cat(
        (
            torch.tensor([0], dtype=torch.int),
            torch.tensor(sizes_q, dtype=torch.int).cumsum(0),
        )
    ).unfold(
        0, 2, 1
    )  # (N_t_q, 2)

    # Initialize the tensors for storing the evaluation metrics one-per-query
    metrics = {key: torch.zeros(N_t_q, dtype=torch.float) for key in _metrics}

    # The query side may live on CPU (pinned) to save GPU memory; move per-batch
    # slices to the DB's device on demand.
    db_device = embeddings_db.device

    # Process batch_size amount of query tracks' segments at once
    for i, (clique_ids_q_batch, version_ids_q_batch, bounds_q) in enumerate(
        zip(
            clique_ids_q.split(batch_size),
            version_ids_q.split(batch_size),
            boundaries_q.split(batch_size),
        )
    ):

        if i == 0 or (i + 1) % 100 == 0 or (i + 1) == (N_t_q // batch_size) + 1:
            print(f"Evaluating batch: {i+1}/{(N_t_q//batch_size)+1}")

        # Embeddings of the query tracks
        Q = embeddings_q[bounds_q[0, 0] : bounds_q[-1, 1]].to(
            db_device, non_blocking=True
        )  # (M, D)
        clique_ids_q_batch = clique_ids_q_batch.to(db_device, non_blocking=True)
        version_ids_q_batch = version_ids_q_batch.to(db_device, non_blocking=True)

        # Number of segments each query track in the batch has
        sizes_q_batch = bounds_q.diff(dim=1).squeeze(1).tolist()  # (B, )

        # Compute the reduced distance matrix one database chunk at a time, then
        # concatenate. Reducing per chunk keeps the intermediate (M, N_c_chunk)
        # matrix bounded by chunk_size.
        D_parts = []
        for ts, te, es, ee in db_chunks:
            db_emb_chunk = embeddings_db[es:ee]
            sizes_db_chunk = sizes_db[ts:te]

            if similarity_search == "MIPS":
                D_part = pairwise_distance_matrix(Q, db_emb_chunk, mode="dot")
            elif similarity_search == "MCSS":
                D_part = pairwise_distance_matrix(Q, db_emb_chunk, mode="cos")
            else:
                # Use the cdist implementation as it is more memory efficient for large batches and databases
                D_part = pairwise_distance_matrix(Q, db_emb_chunk, mode="fro", p=2)
            # D_part: (M, N_c_chunk)

            # Reduce to track-level for this chunk - NOTE: best reduction only
            D_part = reduce_chunk_distances(
                D_part, sizes_q_batch, sizes_db_chunk
            )  # (B, N_chunk)
            D_parts.append(D_part)

        D = D_parts[0] if len(D_parts) == 1 else torch.cat(D_parts, dim=1)  # (B, N)

        # C[i,j] = 1 iff ith q element and jth db element are from the same clique
        C = create_class_matrix(clique_ids_q_batch, clique_ids_db).float()  # (B, N)

        # Set the distance of each query with itself to inf and mark it as non-relevant
        V = create_class_matrix(version_ids_q_batch, version_ids_db)  # (B, N)
        D[V] = float("inf")
        C[V] = 0.0

        # Calculate the metrics for the batch
        batch_metrics = calculate_metrics(D, C, k)
        for key, val in batch_metrics.items():
            metrics[key][i * batch_size : (i + 1) * batch_size] = val

    print(f"Calculation time: {sec_to_hms(t0, True)[1]}")

    # NOTE: in case of multi-gpu processing aggregating here and then averaging the
    # results of each GPU independently will not exactly be equal to gathering all query
    # results and aggregating if the world size do not exactly divide the number of validation
    # loader batches. but the deviation is super small.
    metrics = aggregate_metrics(metrics)

    return metrics
