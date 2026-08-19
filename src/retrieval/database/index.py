import time
from pathlib import Path
from typing import Optional
import warnings

import torch
from cuvs.neighbors import ivf_flat

from src.common.utils import sec_to_hms


def check_finite_memmap(
    database: torch.Tensor, rows_per_chunk: int = 1_000_000
) -> None:
    N = database.shape[0]
    for start in range(0, N, rows_per_chunk):
        end = min(start + rows_per_chunk, N)
        if not torch.isfinite(database[start:end]).all():
            raise AssertionError(f"Database contains NaN/Inf in rows {start}:{end}")


def build_index_and_train(
    database,
    n_lists: int = 1024,
    n_probes: int = 32,
    index_dir: Optional[Path] = None,
):

    assert n_lists >= n_probes > 0

    t0 = time.monotonic()
    print(f"Building the IVF-Flat Index...")

    # To deal with the warning about non-writable NumPy arrays
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        database_torch = torch.as_tensor(
            database, device="cuda", dtype=torch.float16
        ).contiguous()
    check_finite_memmap(database_torch)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    index_params = ivf_flat.IndexParams(n_lists=n_lists, kmeans_trainset_fraction=1.0)
    index = ivf_flat.build(index_params, database_torch)
    search_params = ivf_flat.SearchParams(n_probes=n_probes)
    print(f"Elapsed time: {sec_to_hms(t0, True)[1]}")

    if index_dir is not None:
        save_index(index, index_dir)

    return index, search_params


def load_index(index_dict: dict, index_path: Path) -> tuple:

    index = ivf_flat.load(str(index_path))
    if not index.trained:
        raise RuntimeError(f"Index {index_path} is not trained.")

    search_params = ivf_flat.SearchParams(n_probes=index_dict["n_probes"])

    return index, search_params


def save_index(index: ivf_flat.Index, db_dir: Path):

    db_dir.mkdir(parents=True, exist_ok=True)
    index_path = db_dir / "database.index"
    if index_path.exists:
        print(f"Overwriting existing index.")

    ivf_flat.save(str(index_path), index, include_dataset=True)

    print(f"Index written to {str(index_path)}")
