import time

import torch

from src.common.tensor_op import pairwise_distance_matrix
from src.common.utils import sec_to_hms

from .reduction import reduce_chunk_distances


@torch.inference_mode()
def identify_and_evaluate(
    embeddings_q: torch.Tensor,
    track_ids_q: torch.Tensor,
    sizes_q: list[int],
    embeddings_db: torch.Tensor,
    track_ids_db: torch.Tensor,
    sizes_db: list[int],
    clique_ids_q: torch.Tensor | None = None,
    clique_ids_db: torch.Tensor | None = None,
    similarity_search: str = "MIPS",
    batch_size: int = 1,
) -> dict:
    """Perform track identification: for each query, find the top-k matches in the
    database and check if any share the same track_id. Measures top-1 and top-10
    hit rates at track and (optionally) clique level.

    Parameters:
    -----------
    embeddings_q: torch.Tensor
        2D tensor of shape (m_q, d) where m_q is the total number of segments
        across all query tracks.
    track_ids_q: torch.Tensor
        1D tensor of shape (N_q,) where N_q is the number of query tracks.
    sizes_q: list[int]
        Number of segments per query track.
    embeddings_db: torch.Tensor
        2D tensor of shape (m_db, d) where m_db is the total number of segments
        across all database tracks.
    track_ids_db: torch.Tensor
        1D tensor of shape (N_db,) where N_db is the number of database tracks.
    sizes_db: list[int]
        Number of segments per database track.
    clique_ids_q: torch.Tensor | None
        1D tensor of shape (N_q,) with clique integer IDs for query tracks.
    clique_ids_db: torch.Tensor | None
        1D tensor of shape (N_db,) with clique integer IDs for database tracks.
    similarity_search: str
        Similarity function: "NNS", "MCSS", or "MIPS".
    batch_size: int
        Number of query tracks to process per batch.

    Returns:
    --------
    metrics: dict
        Dictionary containing top-1 and top-10 hit rates and per-query hit tensors.
    """

    t0 = time.monotonic()
    print("Performing track identification...")

    N_t_q = len(sizes_q)
    N_t_db = len(sizes_db)

    assert 1 <= batch_size <= N_t_q

    similarity_search = similarity_search.upper()
    if similarity_search not in ["NNS", "MCSS", "MIPS"]:
        raise ValueError(f"Unknown similarity search: {similarity_search}")

    assert track_ids_q.ndim == 1, f"{track_ids_q.shape}"
    assert embeddings_q.ndim == 2, f"{embeddings_q.shape}"
    assert (
        len(sizes_q) == track_ids_q.shape[0]
    ), f"{len(sizes_q)}, {track_ids_q.shape[0]}"

    assert track_ids_db.ndim == 1, f"{track_ids_db.shape}"
    assert embeddings_db.ndim == 2, f"{embeddings_db.shape}"
    assert (
        len(sizes_db) == track_ids_db.shape[0]
    ), f"{len(sizes_db)}, {track_ids_db.shape[0]}"

    N_c_db = sum(sizes_db)
    assert embeddings_db.shape[0] == N_c_db, "Embeddings and sizes do not match"
    print(f"Number of tracks in the database: {N_t_db:,}")
    print(f"Number of segments in the database: {N_c_db:,}")

    N_c_q = sum(sizes_q)
    assert embeddings_q.shape[0] == N_c_q, "Embeddings and sizes do not match"
    print(f"Number of tracks in the queries: {N_t_q:,}")
    print(f"Number of segments in the queries: {N_c_q:,}")

    # Track boundaries in the embeddings_q tensor
    boundaries_q = torch.cat(
        (
            torch.tensor([0], dtype=torch.int),
            torch.tensor(sizes_q, dtype=torch.int).cumsum(0),
        )
    ).unfold(
        0, 2, 1
    )  # (N_t_q, 2)

    k = min(10, N_t_db)

    use_clique = clique_ids_q is not None and clique_ids_db is not None

    # Per-query hits
    top1_hits = torch.zeros(N_t_q, dtype=torch.float)
    top10_hits = torch.zeros(N_t_q, dtype=torch.float)
    if use_clique:
        clique_top1_hits = torch.zeros(N_t_q, dtype=torch.float)
        clique_top10_hits = torch.zeros(N_t_q, dtype=torch.float)

    batch_iter = zip(track_ids_q.split(batch_size), boundaries_q.split(batch_size))
    if use_clique:
        batch_iter = zip(
            track_ids_q.split(batch_size),
            boundaries_q.split(batch_size),
            clique_ids_q.split(batch_size),
        )

    for i, batch in enumerate(batch_iter):
        if use_clique:
            track_ids_q_batch, bounds_q, clique_ids_q_batch = batch
        else:
            track_ids_q_batch, bounds_q = batch

        if i == 0 or (i + 1) % 100 == 0 or (i + 1) == (N_t_q // batch_size) + 1:
            print(f"Evaluating batch: {i+1}/{(N_t_q//batch_size)+1}")

        # Embeddings of the query tracks in this batch
        Q = embeddings_q[bounds_q[0, 0] : bounds_q[-1, 1]]  # (M, D)

        # Number of segments each query track in the batch has
        sizes_q_batch = bounds_q.diff(dim=1).squeeze(1).tolist()  # (B, )

        # Compute the pairwise distance matrix between the queries and the database
        if similarity_search == "MIPS":
            D = pairwise_distance_matrix(Q, embeddings_db, mode="dot")
        elif similarity_search == "MCSS":
            D = pairwise_distance_matrix(Q, embeddings_db, mode="cos")
        else:
            # Use the cdist implementation as it is more memory efficient for large batches and databases
            D = pairwise_distance_matrix(Q, embeddings_db, mode="fro", p=2)
        # D: (M, N_c_db)

        # Reduce to track-level via best reduction
        D = reduce_chunk_distances(D, sizes_q_batch, sizes_db)  # (B, N_t_db)

        # Top-k indices
        _, topk_indices = torch.topk(D, k, dim=1, largest=False)  # (B, k)
        topk_track_ids = track_ids_db[topk_indices]  # (B, k)

        # Compare against query track_ids
        match = topk_track_ids == track_ids_q_batch.unsqueeze(1)  # (B, k)

        top1_hits[i * batch_size : (i + 1) * batch_size] = match[:, 0].float()
        top10_hits[i * batch_size : (i + 1) * batch_size] = match.any(dim=1).float()

        if use_clique:
            topk_clique_ids = clique_ids_db[topk_indices]  # (B, k)
            clique_match = topk_clique_ids == clique_ids_q_batch.unsqueeze(1)  # (B, k)
            clique_top1_hits[i * batch_size : (i + 1) * batch_size] = clique_match[:, 0].float()
            clique_top10_hits[i * batch_size : (i + 1) * batch_size] = clique_match.any(dim=1).float()

    top1_hit_rate = top1_hits.mean().item()
    top10_hit_rate = top10_hits.mean().item()

    print(f"Top-1 hit rate: {top1_hit_rate:.4f}")
    print(f"Top-10 hit rate: {top10_hit_rate:.4f}")

    metrics = {
        "top1_hit_rate": top1_hit_rate,
        "top10_hit_rate": top10_hit_rate,
        "top1_hits": top1_hits,
        "top10_hits": top10_hits,
    }

    if use_clique:
        clique_top1_hit_rate = clique_top1_hits.mean().item()
        clique_top10_hit_rate = clique_top10_hits.mean().item()
        print(f"Clique top-1 hit rate: {clique_top1_hit_rate:.4f}")
        print(f"Clique top-10 hit rate: {clique_top10_hit_rate:.4f}")
        metrics["clique_top1_hit_rate"] = clique_top1_hit_rate
        metrics["clique_top10_hit_rate"] = clique_top10_hit_rate
        metrics["clique_top1_hits"] = clique_top1_hits
        metrics["clique_top10_hits"] = clique_top10_hits

    print(f"Calculation time: {sec_to_hms(t0, True)[1]}")

    return metrics
