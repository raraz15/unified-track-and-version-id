import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


# TODO: move to src/retrieval/utils.py
def load_emb(
    path_emb: Path | str,
    device: torch.device | str = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor | None, str]:
    path_emb = Path(path_emb)
    fstem = path_emb.stem
    try:
        emb = np.load(path_emb)
        assert emb.dtype in (np.float16, np.float32), (
            f"Embedding {str(path_emb)} must be of dtype float16 or float32."
            f"Got {emb.dtype} instead."
        )
    except Exception as e:
        print(f"Failed to load embedding from {str(path_emb)}:\n{repr(e)}")
        return None, fstem
    emb = torch.as_tensor(emb, device=device, dtype=dtype).contiguous()
    assert torch.isfinite(emb).all(), "Embedding contains NaN or Inf values."
    if emb.ndim != 2:
        print(f"Embedding {str(path_emb)} must be 2D.")
        return None, fstem

    return emb, fstem


def load_query(query_path: Path, gt: dict, fstem2idx: dict, id_level: str):

    # Load the query embedding
    query_emb, q_fstem = load_emb(query_path, device="cuda")

    # Get ground truth information for the query
    gt_entry = gt.get(q_fstem)
    if gt_entry is None:
        print(f"For {str(query_path)} can not find a ground truth entry.")
        return None, q_fstem

    # TODO is the track level necessary?
    # NOTE I tried doing this more elegantly. But it seems that the only way to
    # keep the GT file intact is o skip query files with gt problems.
    if id_level == "track":
        # Get the reference track's index inside the database
        ref_fstem = gt_entry["reference_fstem"]
        ref_track_idx = fstem2idx.get(ref_fstem)
        if ref_track_idx is None:
            print(
                f"Reference track {ref_fstem} for query {str(query_path)} not found in the database."
            )
            return None, q_fstem
        return query_emb, q_fstem
    elif id_level == "version":
        # A query's reference clique must have at least one another version in the database
        if len(gt_entry["valid_other_version_yt_ids"]) < 1:
            print(
                f"Query {str(query_path)} has a reference clique with only {len(gt_entry['valid_other_version_yt_ids'])} other versions."
            )
            return None, q_fstem
        return query_emb, q_fstem
    else:
        raise ValueError(
            "id_level must be either 'track' or 'version'. " f"Got {id_level} instead."
        )


def load_ground_truth(gt_path: Path, id_level: str, db_emb_fnames: list[Path]) -> dict:
    assert gt_path.is_file(), f"Ground truth file {gt_path} does not exist."
    print(f"Loading the ground truth: {gt_path}")

    db_emb_fstems = set([p.stem for p in db_emb_fnames])

    gt = {}
    if id_level == "track":

        # gt_path must be a csv file with the following fields
        # "type", "query_relative_path", "reference_relative_path", "chunk_boundary_start_idx",
        # "chunk_boundary_end_idx", "sample_rate"
        with open(gt_path) as in_f:
            reader = csv.reader(in_f)
            next(reader)
            for row in reader:
                query_fstem = Path(row[1]).stem
                ref_fstem = Path(row[2]).stem
                gt[query_fstem] = {"reference_fstem": ref_fstem}

    # Load the version relationship gt
    elif id_level == "version":

        # Load the version relationship ground truth from a JSON file
        with open(gt_path) as in_f:
            cliques = json.load(in_f)

        # Create a ground truth entry for each version
        for clique_id, versions in cliques.items():
            # Find how many versions of the clique are actually in the database
            yt_ids = {v["youtube_id"] for v in versions}
            yt_ids_valid = yt_ids.intersection(db_emb_fstems)

            for version in versions:

                # in Discogs-VI and SHS, file names are youtube_id.wav
                # SHS has no real version id but we had it assigned in a previous project
                gt[version["youtube_id"]] = {
                    "clique_id": clique_id,
                    "version_id": version["version_id"],
                    "clique_size": len(versions),
                    "valid_other_version_yt_ids": list(
                        yt_ids_valid - {version["youtube_id"]}
                    ),
                }
    else:
        raise ValueError(
            "You must specify either a ground truth file or a version relationship file."
        )

    assert len(gt) > 0, "Ground truth is empty. Please check the input files."
    print(f"Ground truth loaded with {len(gt):,} entries.")

    return gt


def infer_output_dir_from_queries(queries: Path, id_level: str) -> Path:
    """Infers the evaluation output directory from a query embeddings path
    following the new log structure:

        /logs/{ver}/emb/{model}/{dataset}/queries/{granularity}/[manip/...]
    ⟶  /logs/{ver}/eval/{task}/{dataset}/{model}/[granularity]/[manip]

    Args:
        queries (Path): Path to a directory containing query embeddings.
        id_level (str): Identification level, either "track" or "version".

    Returns:
        Path: Inferred output directory path.
    """
    qpath = queries.resolve()
    parts = qpath.parts

    # Map id_level → task segment
    task = "track-id" if id_level == "track" else "version-id"

    try:
        emb_idx = parts.index("emb")
        ver = parts[emb_idx - 1]  # e.g. v0-fp16
        model = parts[emb_idx + 1]  # e.g. nmfp
        dataset = parts[emb_idx + 2]  # e.g. dvi
    except (ValueError, IndexError):
        # Fallback if path doesn't match convention
        return Path.cwd() / "logs" / "eval" / task / id_level

    # Detect granularity (chunks/tracks) and manipulation (rest)
    granularity = None
    manip = None
    try:
        q_idx = parts.index("queries")
        if len(parts) > q_idx + 1:
            granularity = parts[q_idx + 1]
        if len(parts) > q_idx + 2:
            manip = Path(*parts[q_idx + 2 :])
    except ValueError:
        pass

    # Construct target evaluation directory
    root = Path(*parts[: emb_idx - 1]) / ver / "eval" / task / dataset / model
    if granularity:
        root = root / granularity
    if manip:
        root = root / manip

    return root
