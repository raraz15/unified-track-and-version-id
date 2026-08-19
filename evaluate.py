import argparse
import time
import json
from pathlib import Path

import torch
from cuvs.neighbors import ivf_flat

from src.retrieval.utils import find_queries
from src.retrieval.database import get_index
from src.retrieval.reduction import reduce_to_track_level
from src.evaluation.utils import (
    load_ground_truth,
    load_query,
    infer_output_dir_from_queries,
)
import src.evaluation.metrics as metrics
from src.common.utils import sec_to_hms, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "queries",
        type=Path,
        help="""Path to a directory containing the query files. All embeddings
        in a file will be queried, one-by-one.""",
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="""Path to the ground truth CSV file. For track-level identification,
        it should contain pairs of query and reference file names (query_fname, ref_fname).
        For version-level identification, it should contain version and clique information.""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--database-embeddings",
        type=Path,
        default=None,
        help="Path to the database directory where embeddings are individual.",
    )
    group.add_argument(
        "--database-memmap",
        type=Path,
        default=None,
        help="Path to the database directory which had been built previously.",
    )
    group.add_argument(
        "--database-index",
        type=Path,
        default=None,
        help="Path to the directory that contains a trained and populated index.",
    )
    parser.add_argument(
        "--id-level",
        "-i",
        choices=["track", "version"],
        required=True,
        help="Identification level.",
    )
    parser.add_argument(
        "--n-lists",
        type=int,
        default=None,
        help="""Number of lists (clusters) to partition the database. By default uses the smallest 
        positive iteger multiple of 8 of the square root of the database size.""",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=None,
        help="""Number of neighboring cells to visit during the coarse search. By default it 
        uses the 1% of the n-lists for track ID and 3% for version ID.""",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="""Number of top candidate segments to retrieve for each query segment. 
        By default, 1024 for track ID and 10240 for version ID.""",
    )
    parser.add_argument(
        "--top-N",
        "-N",
        type=int,
        default=None,
        help="""Number of top tracks to consider after reduction. By default, all retrieved results 
        are considered.""",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory.",
    )
    args = parser.parse_args()

    set_seed()

    assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."

    index_dict = {
        "n_lists": args.n_lists,
        "n_probes": args.n_probes,
    }
    index, search_params, track_boundaries, db_emb_fnames = get_index(
        id_level=args.id_level,
        index_dict=index_dict,
        index_path=args.database_index,
        merged_emb_path=args.database_memmap,
        embeddings_dir=args.database_embeddings,
        num_workers=args.num_workers,
    )

    if args.top_k is None:
        args.top_k = 1024 if args.id_level == "track" else 10240
        print(f"Setting top-k to {args.top_k}.")

    idx2fstem = {i: p.stem for i, p in enumerate(db_emb_fnames)}
    fstem2idx = {p: i for i, p in idx2fstem.items()}

    # TODO can we clean this up?
    track_boundaries = track_boundaries.to("cuda")
    track_lengths = (track_boundaries[:, 1] - track_boundaries[:, 0]).to(torch.int32)
    track_starts = torch.cat(
        [track_boundaries[:, 0], track_boundaries[-1, 1].view(1)], dim=0
    )  # (N+1,)

    gt = load_ground_truth(args.ground_truth, args.id_level, db_emb_fnames)

    query_paths = find_queries(args.queries)
    display_interval = min(1000, len(query_paths))

    print("Querying one file at a time...")
    t0, t_total = time.monotonic(), 0
    all_queries_results = []
    for i, query_path in enumerate(query_paths):

        # Load the query embedding(s)
        query_emb, q_fstem = load_query(query_path, gt, fstem2idx, args.id_level)
        if query_emb is None:
            continue

        # For each segment in the query sequence, independently retrieve the
        # top-k approximately most similar segments from the database.
        # S: appprox. distance or inner product (len(query_emb), top_k)
        # I: candidate indices matrix (len(query_emb), top_k)
        S, I = ivf_flat.search(search_params, index, query_emb, args.top_k)

        # Reduce the segment-level results to track-level results
        ranking = reduce_to_track_level(S, I, track_lengths, track_starts)

        # Cut off the ranking to the top-N tracks for faster evaluation
        if args.top_N is not None:
            ranking = ranking[: args.top_N]

        # Get the metadata for the results.
        q_results = []
        for d, coord, idx_track in ranking:
            q_results.append(
                {
                    "candidate_fstem": idx2fstem[idx_track],
                    "candidate_segment_local_index": coord[1],
                    "query_segment_local_index": coord[0],
                    "score": d,
                }
            )

        # Evaluate the results for the query
        if args.id_level == "track":
            q_metrics = metrics.track_id_metrics(q_fstem, q_results, gt)
        else:
            q_metrics = metrics.version_id_metrics(
                q_fstem, q_results, gt, len(db_emb_fnames), top_N=args.top_N
            )
        all_queries_results.append(q_metrics)

        # 6 - Display progress
        if i == 0 or ((i + 1) % display_interval) == 0 or (i + 1) == len(query_paths):
            elapsed, elapsed_str = sec_to_hms(t0, True)
            print(f"Queried {i + 1:,}/{len(query_paths):,} files. [{elapsed_str}]")
            t_total += elapsed
            t0 = time.monotonic()
    print(f"Total time: {sec_to_hms(t_total, False)[1]}")

    # Aggregate the results of all individual queries.
    if args.id_level == "track":
        all_queries_results = metrics.aggregate_track_id_metrics(all_queries_results)
    else:
        if args.top_N is not None:
            print(f"Results were cut off at {args.top_N} retrieved tracks.")
        all_queries_results = metrics.aggregate_version_id_metrics(all_queries_results)

    if args.output_dir is None:
        args.output_dir = infer_output_dir_from_queries(args.queries, args.id_level)
    print(f"Output directory: \033[32m{str(args.output_dir)}\033[0m")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"

    # Write the results to disk
    print(f"Metrics are written to \033[32m{metrics_path}\033[0m")
    with open(metrics_path, "w") as o_f:
        json.dump(all_queries_results, o_f, indent=4)
        o_f.write("\n")

    print("Done!")
