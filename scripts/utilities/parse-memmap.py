"""Parse a database memmap into per-track embedding files."""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.database import load_database_memmap


def parse_memmap(
    db_mm_path: Path,
    output_dir: Path,
    delete_original: bool = False,
):

    # Time the parsing
    start_time = time.monotonic()

    # Load the database memmap
    arr, track_boundaries, track_paths = load_database_memmap(db_mm_path)

    # Find the common root directory of the tracks
    common_root = os.path.commonpath(track_paths)
    print(f"Common root directory: {common_root}")

    # Parse the fingerprints
    print(f"=== Parsing the mixture memmap to individual tracks ===")
    for i, (track_path, boundary) in enumerate(zip(track_paths, track_boundaries)):
        # Load the fingerprints of the track from the merged memmap
        fp = np.array(arr[boundary[0] : boundary[1], :])
        # Write the fingerprints of the track to a file
        # NOTE: currently track_path is just the fname
        # fp_path = (output_dir / track_path.relative_to(common_root)).with_suffix(".npy")
        fp_path = output_dir / track_path.stem[:3] / track_path.with_suffix(".npy").name
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fp_path, fp)
        if (i + 1) % 1000 == 0 or i + 1 == len(track_paths):
            print(f"Processed {i + 1:,} / {len(track_paths):,} tracks")

    # Close memmap
    del arr

    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - start_time))
    print(f"Elapsed time for parsing the fingerprints: {elapsed_time}")

    if delete_original:
        print("Deleting the memmap file...")
        os.remove(db_mm_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Parse a memmap file containing fingerprints into individual track files."
    )
    parser.add_argument("mm_path", type=Path, help="Path to the memmap file.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the parsed tracks."
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete the original memmap file after parsing.",
    )
    args = parser.parse_args()

    parse_memmap(
        db_mm_path=args.mm_path,
        output_dir=args.output_dir,
        delete_original=args.delete_original,
    )
