from pathlib import Path


def find_queries(queries_dir: Path) -> list[Path]:

    assert queries_dir.is_dir(), f"Queries directory {queries_dir} does not exist."
    print(f"Finding the query files in {queries_dir}")
    query_paths = sorted(list(queries_dir.rglob("*.npy")))
    print(f"Found {len(query_paths):,} query files.")
    assert len(query_paths) > 0, f"No query files found in {queries_dir}."

    return query_paths
