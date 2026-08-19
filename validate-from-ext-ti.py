import copy
from pathlib import Path
import argparse
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.fish.validation.retrieval_tracks import identify_and_evaluate

# Metrics kept for the final evaluation output
EVAL_METRICS = {
    "top1_hit_rate",
    "top10_hit_rate",
    "clique_top1_hit_rate",
    "clique_top10_hit_rate",
}


def load_db_embs(
    emb_dir: Path,
    cliques: dict,
    workers: int = 6,
    float16: bool = True,
):
    """Load database embeddings, pruning missing versions/cliques. Returns db_yt_ids."""

    print("Loading the database embeddings...")

    cliques = copy.deepcopy(cliques)

    db_yt_ids = set()
    for clique_id in list(cliques.keys()):
        delete = []
        for i in range(len(cliques[clique_id])):
            yt_id = cliques[clique_id][i]["youtube_id"]
            emb_path = emb_dir / yt_id[:2] / f"{yt_id}.npy"
            if not emb_path.exists():
                delete.append(i)
                continue
            cliques[clique_id][i]["emb_path"] = emb_path
            db_yt_ids.add(yt_id)
        for i in reversed(delete):
            del cliques[clique_id][i]
        if len(cliques[clique_id]) < 2:
            del cliques[clique_id]

    items = [
        (clique_id, version)
        for clique_id, versions in cliques.items()
        for version in versions
    ]

    clique_id_map = {cid: i for i, cid in enumerate(cliques.keys())}

    def _load(item):
        clique_id, version = item
        try:
            emb = torch.from_numpy(np.load(version["emb_path"]))
        except (EOFError, ValueError, OSError) as e:
            print(f"Skipping {version['emb_path']}: {e}")
            return None
        if float16:
            emb = emb.half()
        track_id = int(version["version_id"].split("-")[1])
        return (emb, track_id, clique_id_map[clique_id])

    embs, track_ids, clique_ids, sizes = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_load, items):
            if result is None:
                continue
            emb, track_id, clique_id = result
            embs.append(emb)
            sizes.append(emb.shape[0])
            track_ids.append(track_id)
            clique_ids.append(clique_id)

    embs = torch.cat(embs, dim=0).to("cuda")
    track_ids = torch.tensor(track_ids).to("cuda")
    clique_ids = torch.tensor(clique_ids).to("cuda")
    return embs, track_ids, clique_ids, sizes, db_yt_ids, clique_id_map


def load_query_embs(
    emb_dir: Path,
    cliques: dict,
    db_yt_ids: set,
    clique_id_map: dict,
    workers: int = 6,
    float16: bool = True,
):
    """Load query embeddings, skipping versions not present in the database."""

    print("Loading the query embeddings...")

    cliques = copy.deepcopy(cliques)

    for clique_id in list(cliques.keys()):
        delete = []
        for i in range(len(cliques[clique_id])):
            yt_id = cliques[clique_id][i]["youtube_id"]
            if yt_id not in db_yt_ids:
                delete.append(i)
                continue
            emb_path = emb_dir / yt_id[:2] / f"{yt_id}.npy"
            if not emb_path.exists():
                delete.append(i)
                continue
            cliques[clique_id][i]["emb_path"] = emb_path
        for i in reversed(delete):
            del cliques[clique_id][i]
        if len(cliques[clique_id]) < 2:
            del cliques[clique_id]

    items = [
        (clique_id, version)
        for clique_id, versions in cliques.items()
        for version in versions
    ]

    def _load(item):
        clique_id, version = item
        try:
            emb = torch.from_numpy(np.load(version["emb_path"]))
        except (EOFError, ValueError, OSError) as e:
            print(f"Skipping {version['emb_path']}: {e}")
            return None
        if float16:
            emb = emb.half()
        track_id = int(version["version_id"].split("-")[1])
        return (emb, track_id, clique_id_map[clique_id])

    embs, track_ids, clique_ids, sizes = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_load, items):
            if result is None:
                continue
            emb, track_id, clique_id = result
            embs.append(emb)
            sizes.append(emb.shape[0])
            track_ids.append(track_id)
            clique_ids.append(clique_id)

    embs = torch.cat(embs, dim=0).to("cuda")
    track_ids = torch.tensor(track_ids).to("cuda")
    clique_ids = torch.tensor(clique_ids).to("cuda")
    return embs, track_ids, clique_ids, sizes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "database_dir",
        type=Path,
        help="Path to the directory containing the database embeddings.",
    )
    parser.add_argument(
        "cliques_json_path",
        type=Path,
        help="Path to the JSON file defining cliques.",
    )
    parser.add_argument(
        "queries", type=Path, help="Path to a directory containing query embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for similarity search.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of workers for parallel embedding loading.",
    )
    parser.add_argument(
        "--similarity-search",
        type=str,
        default="NNS",
        choices=["NNS", "MIPS", "MCSS"],
        help="Similarity search method to use.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory.",
    )
    args = parser.parse_args()

    print(f"Loading cliques from \033[34m{args.cliques_json_path}\033[0m")
    with open(args.cliques_json_path) as f:
        cliques = json.load(f)

    db_embs, db_track_ids, db_clique_ids, db_sizes, db_yt_ids, clique_id_map = (
        load_db_embs(
            args.database_dir,
            cliques,
            workers=args.workers,
        )
    )

    q_embs, q_track_ids, q_clique_ids, q_sizes = load_query_embs(
        args.queries,
        cliques,
        db_yt_ids,
        clique_id_map,
        workers=args.workers,
    )

    metrics = identify_and_evaluate(
        q_embs,
        q_track_ids,
        q_sizes,
        db_embs,
        db_track_ids,
        db_sizes,
        clique_ids_q=q_clique_ids,
        clique_ids_db=db_clique_ids,
        batch_size=args.batch_size,
        similarity_search=args.similarity_search,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("logs/eval") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {k: v for k, v in metrics.items() if k in EVAL_METRICS}

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to \033[34m{metrics_path}\033[0m")

    args_path = output_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    print(f"Wrote arguments to \033[34m{args_path}\033[0m")
