import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .index import load_index, build_index_and_train
from .merge_embeddings import merge_embeddings_to_memmap


def get_index(
    id_level: str,
    index_dict: dict,
    index_path: Optional[Path] = None,
    merged_emb_path: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
    num_workers: int = 6,
) -> tuple:

    if index_path is not None:
        index, search_params = load_index(index_dict, index_path)
        track_boundaries, emb_fnames, _ = load_database_metadata(index_path.parent)
    else:
        if merged_emb_path is not None:
            mm, track_boundaries, emb_fnames = load_database_memmap(merged_emb_path)
            # index_save_dir = merged_emb_path.parent
        elif embeddings_dir is not None:
            mm_path, _ = merge_embeddings_to_memmap(
                embeddings_dir, num_workers=num_workers
            )
            mm, track_boundaries, emb_fnames = load_database_memmap(mm_path)
            # index_save_dir = mm_path.parent
        else:
            raise ValueError(
                "Either index_path, merged_emb_path, or embeddings_dir must be provided."
            )

        # Set default index parameters if not provided
        if index_dict["n_lists"] is None:
            index_dict["n_lists"] = int((math.sqrt(track_boundaries[-1, -1]) // 8) * 8)
            index_dict["n_lists"] = max(8, index_dict["n_lists"])
            print(f"Setting n-lists to {index_dict['n_lists']}")
        if index_dict["n_probes"] is None:
            if id_level == "track":
                index_dict["n_probes"] = int(index_dict["n_lists"] * 1 / 100)
            elif id_level == "version":
                index_dict["n_probes"] = int(index_dict["n_lists"] * 3 / 100)
            else:
                raise NotImplementedError
            index_dict["n_probes"] = max(1, index_dict["n_probes"])
            print(f"Setting n-probes to {index_dict['n_probes']}")
        index, search_params = build_index_and_train(
            database=mm,
            # index_dir=index_save_dir,
            **index_dict,
        )

    return index, search_params, track_boundaries, emb_fnames


def load_database_metadata(database_dir: Path) -> tuple[torch.Tensor, list[Path], int]:
    assert database_dir.is_dir(), f"Database directory {database_dir} does not exist."
    db_csv_path = database_dir / "database.csv"
    assert db_csv_path.exists(), f"Database CSV {db_csv_path} does not exist."

    # Load the CSV file to get the paths and boundaries
    emb_fnames, track_boundaries = [], []
    with open(db_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            emb_fnames.append(Path(row[0]))  # NOTE: its a file name and path
            track_boundaries.append((int(row[1]), int(row[2])))
    emb_dim = int(row[3])

    track_boundaries = torch.as_tensor(
        track_boundaries, dtype=torch.int64, device="cpu"
    )  # (N,2)

    return track_boundaries, emb_fnames, emb_dim


def load_database_memmap(
    db_mm_path: Path,
) -> tuple[np.memmap, torch.Tensor, list[Path]]:

    print("Using existing database...")
    assert db_mm_path.exists(), f"Database {db_mm_path} does not exist."
    assert db_mm_path.suffix == ".mm", f"Database {db_mm_path} is not a memmap file."

    track_boundaries, emb_fnames, emb_dim = load_database_metadata(db_mm_path.parent)

    total = int(track_boundaries[-1, 1].item())
    db_mm = np.memmap(db_mm_path, dtype=np.float16, mode="r", shape=(total, emb_dim))

    print(f"Loaded merged database embeddings from {db_mm_path}")
    print(f"Number of embeddings: {track_boundaries[-1, 1]:,}")
    print(f"Number of tracks: {len(track_boundaries):,}")

    return db_mm, track_boundaries, emb_fnames
