import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import csv
import json
from collections import defaultdict
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.utils import get_git_commit, json_default

# The fields to use from the similar-segments.csv and their dtypes
COLUMN_DTYPES = {
    "clique_id": "object",
    "version_id0": "object",
    "version_id1": "object",
    "youtube_id0": "string",
    "youtube_id1": "string",
    "duration0": "float32",
    "duration1": "float32",
    "segment_duration": "float32",
    "hole_v0_start": "float32",
    "hole_v1_start": "float32",
}

# The fields to write to the output CSV and their order
CSV_FIELDNAMES = [
    "track_clique_id",
    "segment_clique_id",
    "version_id",
    "youtube_id",
    "segment_start_time",
    "segment_end_time",
    "track_duration",
]


class Interval:

    def __init__(self, start: float, end: float):

        if end <= start:
            raise ValueError(f"end ({end}) must be > start ({start})")
        self.start = start
        self.end = end

    def overlaps(self, other):
        return self.end > other.start and self.start < other.end

    def union(self, other):
        # Union was a bad idea...
        raise NotImplementedError
        # if not self.overlaps(other):
        #     raise ValueError
        # return Interval(min(self.start, other.start), max(self.end, other.end))

    def intersection(self, other):
        if not self.overlaps(other):
            raise ValueError
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end <= start:
            raise ValueError
        return Interval(start, end)

    @property
    def length(self):
        return self.end - self.start


def process_clique(rows: pd.DataFrame, merge_condition_dur: float):
    if rows.empty:
        return None

    clique_id_track = str(rows.iloc[0]["clique_id"])

    # Parse holes
    holes = []
    vid_yt_id = {}
    duration = {}
    for row in rows.itertuples(index=False):
        hole = {
            "version_id0": row.version_id0,
            "version_id1": row.version_id1,
            "hole_v0": Interval(float(row.hole_v0_start), float(row.hole_v0_start) + float(row.segment_duration)),  # type: ignore
            "hole_v1": Interval(float(row.hole_v1_start), float(row.hole_v1_start) + float(row.segment_duration)),  # type: ignore
        }
        # Record version to YouTube ID mapping
        if row.version_id0 not in vid_yt_id:
            vid_yt_id[row.version_id0] = row.youtube_id0
        if row.version_id1 not in vid_yt_id:
            vid_yt_id[row.version_id1] = row.youtube_id1
        # Record version durations
        if row.version_id0 not in duration:
            duration[row.version_id0] = float(row.duration0)  # type: ignore
        if row.version_id1 not in duration:
            duration[row.version_id1] = float(row.duration1)  # type: ignore
        holes.append(hole)

    # Group holes into segment cliques based on the merge condition
    hole_groups = build_hole_groups(holes, merge_condition_dur=merge_condition_dur)

    # For each group of holes, create a segment clique with the corresponding segments from each hole
    segment_cliques = []
    for group_indices in hole_groups:
        clique = []
        for idx in group_indices:
            h = holes[idx]
            clique.append(
                {
                    "version_id": h["version_id0"],
                    "youtube_id": vid_yt_id[h["version_id0"]],
                    "segment_start_time": h["hole_v0"].start,
                    "segment_end_time": h["hole_v0"].end,
                    "track_duration": duration[h["version_id0"]],
                }
            )
            clique.append(
                {
                    "version_id": h["version_id1"],
                    "youtube_id": vid_yt_id[h["version_id1"]],
                    "segment_start_time": h["hole_v1"].start,
                    "segment_end_time": h["hole_v1"].end,
                    "track_duration": duration[h["version_id1"]],
                }
            )

        # De-duplicate if the same (version, start, end) exists twice in a group
        # NOTE: this will keep the first occurrence, so some of the information of the second hole,
        # which can be different, will be lost.
        unique_segments = list(
            {
                (s["version_id"], s["segment_start_time"], s["segment_end_time"]): s
                for s in clique
            }.values()
        )
        segment_cliques.append(unique_segments)

    return clique_id_track, segment_cliques


def build_hole_groups(holes: List[dict], merge_condition_dur: float) -> List[List[int]]:
    """Return list of groups, each is a list of indices into holes."""

    n = len(holes)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # 1. Organize all intervals by version_id
    # key: version_id, value: list of (Interval, hole_index)
    intervals_by_version = defaultdict(list)
    for i, hole in enumerate(holes):
        intervals_by_version[hole["version_id0"]].append((hole["hole_v0"], i))
        intervals_by_version[hole["version_id1"]].append((hole["hole_v1"], i))

    # 2. For each version, check overlaps
    for items in intervals_by_version.values():
        if not items:
            continue

        # Sort by start time to enable efficient window search
        items.sort(key=lambda x: x[0].start)

        # Compare each item only with the subsequent items that physically overlap
        for i in range(len(items)):
            iv_a, idx_a = items[i]
            for j in range(i + 1, len(items)):
                iv_b, idx_b = items[j]

                # Since list is sorted by start time, if iv_b starts after iv_a ends,
                # then iv_b AND ALL SUBSEQUENT items cannot overlap iv_a.
                if iv_b.start >= iv_a.end:
                    break

                # They geometrically overlap. Check the overlap condition
                inter_len = iv_a.intersection(iv_b).length
                if inter_len >= merge_condition_dur:
                    union(idx_a, idx_b)

    # 3. Collect connected components
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return list(groups.values())


def handle_result(
    result, writer: csv.DictWriter, next_segment_id: int, total_intervals: int
) -> Tuple[int, int]:
    """Write segment cliques to a single CSV; return next available ID and interval total."""
    if result is None:
        return next_segment_id, total_intervals
    clique_id_track, segment_cliques = result
    csv_rows = []
    for segment_clique in segment_cliques:
        segment_clique_id_str = f"S-{str(next_segment_id).zfill(7)}"
        for segment in segment_clique:
            csv_rows.append(
                {
                    "track_clique_id": clique_id_track,
                    "segment_clique_id": segment_clique_id_str,
                    **segment,
                }
            )
        total_intervals += len(segment_clique)
        next_segment_id += 1
    if csv_rows:
        writer.writerows(csv_rows)
    return next_segment_id, total_intervals


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Find segment-level cliques from tracks-level cliques."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the similar-regions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="""Path to the output directory. The output will be saved in a 
        subdirectory named after the segment-level clique ID.""",
    )
    parser.add_argument(
        "--merge-condition-dur",
        type=float,
        default=19.0,
        help="Minimum overlap duration (in seconds) between holes to merge them into a clique.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers to use.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, usecols=list(COLUMN_DTYPES.keys()), dtype=COLUMN_DTYPES)  # type: ignore
    print(f"Total similar region pairs: {len(df):,}")

    if args.output_dir is None:
        args.output_dir = args.csv_path.parent
        print(f"No output dir specified. Using {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = args.output_dir / "segment-cliques.csv"
    print(f"Output CSV path: {str(output_csv_path)}")

    args_dict = dict(vars(args))
    del args_dict["csv_path"]
    args_dict["input_path"] = str(args.csv_path)
    args_dict["output_path"] = str(output_csv_path)
    args_dict["git_commit"] = get_git_commit()
    args_dict["run_date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args_json_path = args.output_dir / "parameters-segment-cliques.json"
    with open(args_json_path, mode="w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, default=json_default)
    print(f"Saved CLI arguments to {args_json_path}")

    segment_clique_id = 0
    total_intervals = 0
    with open(output_csv_path, mode="w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_clique,
                    group_df,
                    merge_condition_dur=args.merge_condition_dur,
                )
                # Process larger cliques first to process faster
                for _, group_df in sorted(
                    df.groupby("clique_id"), key=lambda x: len(x[1]), reverse=True
                )
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                segment_clique_id, total_intervals = handle_result(
                    fut.result(), writer, segment_clique_id, total_intervals
                )
                if i % 1000 == 0 or i == len(futures):
                    print(
                        f"Processed {i:,} / {len(futures):,} track cliques. {segment_clique_id:,} segment cliques found. {total_intervals:,} regions written.",
                        flush=True,
                    )

    print(f"Total segment-level cliques created: {segment_clique_id:,}.")
    print(f"Total regions written: {total_intervals:,}.")
    print(f"Wrote segment cliques to {output_csv_path}.")
