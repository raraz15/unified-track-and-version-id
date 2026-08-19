"""Takes a query embedding file and a database of embeddings (in a root directory)
and performs exhaustive retrieval, saving the results to a CSV file."""

import argparse
import shutil
from pathlib import Path
import csv
import time
from datetime import datetime
from typing import Tuple

import torch

from src.common.utils import sec_to_hms, set_seed
from src.evaluation.utils import load_emb
from src.common.tensor_op import pairwise_distance_matrix
from src.common.audio import load_audio, write_audio, SAMPLE_RATE

HEADERS = [
    "query",
    "query-segment-index",  # the index of the segment within the query file
    "query-sequence-length",  # the length of the query segment in segments
    "candidate",
    "candidate-segment-index",  # the index of the segment within the retrieved file
    "candidate-sequence-length",  # the length of the candidate segment in segments
    "rank",
    "similarity-score",
]


def find_queries(queries_dir: Path) -> list[Path]:

    if queries_dir.is_dir():
        query_paths = sorted(list(queries_dir.rglob("*.npy")))
        assert len(query_paths) > 0, f"No query files found in {queries_dir}"
        print(f"[Queries] Found {len(query_paths):,} files in {queries_dir}")
    elif queries_dir.is_file():
        print(f"[Queries] Single file provided: {queries_dir}")
        assert (
            queries_dir.suffix == ".npy"
        ), f"Query file {queries_dir} is not a .npy file."
        query_paths = [queries_dir]
    else:
        raise ValueError(
            f"Provided queries path {queries_dir} is neither a .npy file nor a directory."
        )

    return query_paths


# TODO: muli-threaded loading?
def load_database_embeddings(
    embedding_dir: Path | str, device: torch.device | str = torch.device("cuda")
) -> Tuple[torch.Tensor, list[str], list[int]]:

    db_emb_files = sorted(Path(embedding_dir).rglob("*.npy"))
    print(f"[Database] Found {len(db_emb_files):,} embedding files in {embedding_dir}")
    db_embeddings, db_fstems, db_emb_lengths = [], [], []
    for db_emb_file in db_emb_files:
        emb, fstem = load_emb(db_emb_file, device=device)
        if emb is None:
            continue
        db_embeddings.append(emb)
        db_fstems.append(fstem)
        db_emb_lengths.append(emb.shape[0])
    assert len(db_embeddings) > 0, "No database embeddings were loaded."
    db_embeddings = torch.cat(db_embeddings, dim=0)  # (N_db, D)
    print(f"[Database] Embeddings shape: {db_embeddings.shape}")
    return db_embeddings, db_fstems, db_emb_lengths


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
        "database_embeddings",
        type=Path,
        help="Path to the database directory where individual embeddings are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            f"logs/retrieval-results/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ),
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--top-N",
        "-N",
        type=int,
        default=20,
        help="""Number of top tracks to consider after reduction. By default, all retrieved results 
        are considered.""",
    )
    parser.add_argument(
        "--similarity-measure",
        choices=["MCSS", "NNS", "MIPS"],
        default="MIPS",
        help="Similarity measure to use.",
    )
    parser.add_argument(
        "--audio-dir-database",
        type=Path,
        default=None,
        help="If you provide this, the script will write the candidate audio segments to disk.",
    )
    parser.add_argument(
        "--audio-dir-queries",
        type=Path,
        default=None,
        help="If you provide this, the script will write the query audio to disk.",
    )
    parser.add_argument(
        "--segment-dur",
        type=float,
        default=20.0,
        help="Duration of each embedding segment in seconds.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.75,
        help="Overlap ratio between consecutive segments (e.g. 0.75 = 75%% overlap).",
    )
    args = parser.parse_args()

    hop_dur = args.segment_dur * (1 - args.overlap_ratio)

    set_seed()

    assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."
    assert args.top_N >= 1, f"top-N must be at least 1, got {args.top_N}"

    embeddings_db, fstems_db, sizes_db = load_database_embeddings(
        args.database_embeddings, device="cuda"
    )  # (N_db, D)
    track_boundaries_db = torch.tensor(
        [0] + sizes_db, dtype=torch.int64, device="cuda"
    ).cumsum(dim=0)

    query_paths = find_queries(args.queries)

    print(f"Saving retrieval results to: {str(args.output_dir.resolve())}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []

    t0 = time.monotonic()
    # TODO support multi-query?
    for i, q_emb_path in enumerate(query_paths):

        # emb_query: (N_q, D)
        emb_query, fstem_q = load_emb(q_emb_path, device="cuda")
        if emb_query is None:
            continue

        # TODO make this better
        csv_path = args.output_dir / q_emb_path.parent.name / fstem_q / f"{fstem_q}.csv"
        csv_paths.append(csv_path)
        output_q_dir = csv_path.parent
        output_q_dir.mkdir(parents=True, exist_ok=True)

        # TODO: chunk the queries if too large to fit in memory
        # Compute the pairwise similarity matrix between the queries and the database
        # (N_s_q, N_s_db)
        if args.similarity_measure == "MIPS":
            S = pairwise_distance_matrix(emb_query, embeddings_db, mode="dot")
        elif args.similarity_measure == "MCSS":
            S = pairwise_distance_matrix(emb_query, embeddings_db, mode="cos")
        else:
            S = pairwise_distance_matrix(emb_query, embeddings_db, mode="fro", p=2)

        # For each query embedding, return the top-N most similar database embeddings
        sim_scores, db_segment_indices = torch.topk(
            S, args.top_N, dim=1, largest=False
        )  # (N_s_q, top-N)
        # Map segment indices to track indices
        db_track_indices = (
            torch.bucketize(db_segment_indices, track_boundaries_db, right=True) - 1
        )  # (N_s_q, top-N)
        sim_scores = sim_scores.cpu().tolist()
        db_segment_indices = db_segment_indices.cpu().tolist()
        db_track_indices = db_track_indices.cpu().tolist()

        with csv_path.open("w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(HEADERS)

            # For each query segment's results
            for k, (sim_score, db_seg_ind, db_track_ind) in enumerate(
                zip(sim_scores, db_segment_indices, db_track_indices)
            ):
                # Find the local segment index within each candidate track
                cand_seg_indices = [
                    db_seg_ind[j] - track_boundaries_db[track_ind].cpu().item()
                    for j, track_ind in enumerate(db_track_ind)
                ]
                # For each candidate segment
                for l, (score, seg_ind_c, track_ind_c) in enumerate(
                    zip(sim_score, cand_seg_indices, db_track_ind)
                ):
                    size_c = sizes_db[track_ind_c]
                    fstem_c = fstems_db[track_ind_c]  # Corresponding file name
                    csv_writer.writerow(
                        [
                            fstem_q,
                            k,
                            emb_query.shape[0],
                            fstem_c,
                            seg_ind_c,
                            size_c,
                            l,
                            f"{score:.5f}",
                        ]
                    )
                    if args.audio_dir_database is not None:
                        cand_audio_path = (
                            args.audio_dir_database / fstem_c[:2] / f"{fstem_c}.wav"
                        )
                        segment = load_audio(
                            cand_audio_path,
                            start=int(seg_ind_c * SAMPLE_RATE * hop_dur),
                            length=int(SAMPLE_RATE * args.segment_dur),
                        )
                        out_path = output_q_dir / f"{l+1:03d}-{fstem_c}-{seg_ind_c}.wav"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        write_audio(out_path, segment, SAMPLE_RATE)
                    if args.audio_dir_queries is not None:
                        q_audio_path = (
                            args.audio_dir_queries / fstem_q[:2] / f"{fstem_q}.wav"
                        )
                        shutil.copy(
                            q_audio_path,
                            output_q_dir,
                        )

        if (i == 0) or (i + 1 % 10) or (i + 1) == len(query_paths):
            print(f"[{i+1}/{len(query_paths)}] Elapsed time: {sec_to_hms(t0, True)[1]}")

    for csv_path in csv_paths:
        print(f"[Results] {csv_path}")
        with csv_path.open("r", newline="") as result_file:
            for row in csv.reader(result_file):
                print(",".join(row))

    print("Done!")
