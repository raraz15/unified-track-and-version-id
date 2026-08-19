from pathlib import Path
import argparse
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.fish.validation.retrieval_versions import retrieve_and_evaluate


def load_query_embs(
    emb_dir: Path,
    cliques: dict,
    db_clique_ids: set,
    workers: int = 6,
    float16: bool = True,
):
    """Load query embeddings, filtering by database state.

    A query version is valid if its clique has 2+ versions in the database
    (clique_id in db_clique_ids), regardless of how many other query versions
    share the same clique.
    """
    items = []
    for clique_id, versions in cliques.items():
        if clique_id not in db_clique_ids:
            continue
        for version in versions:
            yt_id = version["youtube_id"]
            emb_path = emb_dir / yt_id[:2] / f"{yt_id}.npy"
            if emb_path.exists():
                items.append((clique_id, {**version, "emb_path": emb_path}))

    def _load(item):
        clique_id, version = item
        emb = torch.from_numpy(np.load(version["emb_path"]))
        if float16:
            emb = emb.half()
        return (
            emb,
            int(clique_id.split("-")[1]),
            int(version["version_id"].split("-")[1]),
        )

    embs, c_ids, v_ids, sizes = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_load, items):
            emb, c_id, v_id = result
            embs.append(emb)
            sizes.append(emb.shape[0])
            c_ids.append(c_id)
            v_ids.append(v_id)

    # If the queries and database don't fit to VRAM together,
    # you can load them one-by-one, or load all and send to CUDA only the batch size
    embs = torch.cat(embs, dim=0).to("cuda")
    c_ids = torch.tensor(c_ids).to("cuda")
    v_ids = torch.tensor(v_ids).to("cuda")
    return embs, c_ids, v_ids, sizes


def load_db_embs(
    emb_dir: Path,
    cliques: dict,
    workers: int = 6,
    float16: bool = True,
):
    """Load embeddings from emb_dir, pruning missing versions/cliques in-place."""
    for clique_id in list(cliques.keys()):
        delete = []
        for i in range(len(cliques[clique_id])):
            yt_id = cliques[clique_id][i]["youtube_id"]
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
        emb = torch.from_numpy(np.load(version["emb_path"]))
        if float16:
            emb = emb.half()
        return (
            emb,
            int(clique_id.split("-")[1]),
            int(version["version_id"].split("-")[1]),
        )

    embs, c_ids, v_ids, sizes = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_load, items):
            emb, c_id, v_id = result
            embs.append(emb)
            sizes.append(emb.shape[0])
            c_ids.append(c_id)
            v_ids.append(v_id)

    embs = torch.cat(embs, dim=0).to("cuda")
    c_ids = torch.tensor(c_ids).to("cuda")
    v_ids = torch.tensor(v_ids).to("cuda")
    return embs, c_ids, v_ids, sizes


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
        help="Path to the JSON file defining cliques (version groupings).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to a separate directory containing query embeddings. "
        "If not provided, queries are drawn from the database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (number of query files) for similarity search.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (number of database tracks) for similarity search.",
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

    # Keep the original cliques for query loading (query versions may not be in the DB).
    original_cliques = {k: [dict(v) for v in vs] for k, vs in cliques.items()}

    print("Loading the database embeddings...")
    db_embs, db_c_ids, db_v_ids, db_sizes = load_db_embs(
        args.database_dir,
        cliques,
        workers=args.workers,
    )

    if args.queries is not None:
        print("Loading query embeddings...")
        # A query is valid if its clique has 2+ versions in the database — checked via
        # db_clique_ids, not by counting how many query versions share the clique.
        db_clique_ids = set(cliques.keys())
        q_embs, q_c_ids, q_v_ids, q_sizes = load_query_embs(
            args.queries,
            original_cliques,
            db_clique_ids,
            workers=args.workers,
        )
    else:
        q_embs, q_c_ids, q_v_ids, q_sizes = db_embs, db_c_ids, db_v_ids, db_sizes

    metrics = retrieve_and_evaluate(
        q_embs,
        q_c_ids,
        q_v_ids,
        q_sizes,
        db_embs,
        db_c_ids,
        db_v_ids,
        db_sizes,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        similarity_search=args.similarity_search,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("logs/eval") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to \033[34m{metrics_path}\033[0m")

    args_path = output_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    print(f"Wrote arguments to \033[34m{args_path}\033[0m")
