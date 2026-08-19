import sys
import json
import csv
import time
import argparse
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.audio import SAMPLE_RATE
from src.common.utils import s_to_hms, get_git_commit
from src.similar_region.functions import process_clique
from src.similar_region.field_names import FIELD_NAMES

fmt = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

# 2) stdout handler for INFO
h_info = logging.StreamHandler(sys.stdout)
h_info.setLevel(logging.INFO)
h_info.addFilter(lambda r: r.levelno <= logging.INFO)
h_info.setFormatter(fmt)
logger.addHandler(h_info)

# 3) stderr handler for WARNING+
h_err = logging.StreamHandler(sys.stderr)
h_err.setLevel(logging.WARNING)
h_err.setFormatter(fmt)
logger.addHandler(h_err)


def _process_clique_worker(task):
    clique_id, clique, cfg = task
    try:
        rows = process_clique(
            clique_id=clique_id,
            clique=clique,
            **cfg,
            logger=logger,
        )
    except Exception:
        logger.exception("Error processing clique %s", clique_id)
        rows = []
    return clique_id, rows


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process a clique of versions and write the results to a CSV file."
    )
    parser.add_argument(
        "discogs_vi_yt_path",
        type=Path,
        help="Path to Discogs-VI-YT-light.json",
    )
    parser.add_argument(
        "music_dir",
        type=Path,
        help="Path to the directory containing the music files.",
    )
    parser.add_argument(
        "embeddings_dir",
        type=Path,
        help="Path to the directory containing the embeddings.",
    )
    parser.add_argument(
        "non_music_annotations_dir",
        type=Path,
        help="Path to the directory containing non-music annotations.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="""Path to the output directory. A single CSV file is saved under here.""",
    )
    parser.add_argument(
        "--shingle-dur",
        default=20.0,
        type=float,
        help="Duration of the shingle in seconds.",
    )
    parser.add_argument(
        "--shingle-hop-dur",
        default=1.0,
        type=float,
        help="Hop duration of the shingle in seconds.",
    )
    parser.add_argument(
        "--max-basins",
        default=None,
        type=int,
        help="Maximum number of basins to find for a version pair.",
    )
    parser.add_argument(
        "--global-pct-threshold",
        default=5.0,
        type=float,
        help="Global percentile limit for the basin.",
    )
    parser.add_argument(
        "--universal-dist-threshold",
        default=2.0,
        type=float,
        help="""Maximum height to consider for basin finding. Used to filter false positives. 
        We empirically found this value.""",
    )
    parser.add_argument(
        "--removal-type",
        type=str,
        default="rectangle",
        choices=["cross", "basin", "rectangle"],
        help=("How to mask a discovered basin before searching for the next one."),
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-30.0,
        help="Threshold in dB to consider a frame as silence.",
    )
    parser.add_argument(
        "--silence-min-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of silent frames in a shingle to consider it silent.",
    )
    parser.add_argument(
        "--max-basins-write",
        default=50,
        type=int,
        help="Maximum number of basins to write per version pair.",
    )
    parser.add_argument(
        "--num-workers",
        default=40,
        type=int,
        help="Number of worker processes (defaults to CPU count or number of cliques).",
    )
    args = parser.parse_args()

    assert (
        args.shingle_dur >= 10.0
    ), "Shingle duration must be at least 10 seconds to align with non-music annotations."
    assert (
        args.shingle_hop_dur == 1.0
    ), "Shingle hop duration must be 1 second to align with non-music annotations."
    assert (
        args.non_music_annotations_dir.exists()
    ), f"Non-music annotations directory {args.non_music_annotations_dir} does not exist."

    # Load the cliques
    with open(args.discogs_vi_yt_path) as in_f:
        discogs_vi_yt = json.load(in_f)
    clique_ids = list(discogs_vi_yt.keys())
    logger.info("%d cliques found", len(clique_ids))

    # Lexicographic sort the cliques for better load balancing
    clique_ids = sorted(
        clique_ids,
        key=lambda cid: (len(discogs_vi_yt[cid]), cid),
        reverse=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = args.output_dir / "similar-regions.csv"
    logger.info("Output CSV path: %s", str(output_csv_path))

    # Save the parameters
    logger.info("Arguments: %s", vars(args))
    params = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    params["csv_path"] = str(output_csv_path)
    params["run_date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params["git_commit"] = get_git_commit()
    with open(args.output_dir / "parameters-similar-regions.json", "w") as out_f:
        json.dump(params, out_f, indent=4)

    worker_cfg = {
        "shingle_dur": args.shingle_dur,
        "shingle_hop_dur": args.shingle_hop_dur,
        "sample_rate": SAMPLE_RATE,
        "embeddings_dir": Path(args.embeddings_dir),
        "silence_threshold_db": args.silence_threshold_db,
        "silence_min_fraction": args.silence_min_fraction,
        "music_dir": Path(args.music_dir),
        "non_music_annotations_dir": Path(args.non_music_annotations_dir),
        "global_pct_threshold": args.global_pct_threshold,
        "max_basins": args.max_basins,
        "max_basins_write": args.max_basins_write,
        "universal_dist_threshold": args.universal_dist_threshold,
        "remove_method": args.removal_type,
    }
    tasks = [
        (clique_id, discogs_vi_yt[clique_id], worker_cfg) for clique_id in clique_ids
    ]

    output_csv_file = open(output_csv_path, mode="w", newline="")
    output_csv_writer = csv.DictWriter(output_csv_file, fieldnames=FIELD_NAMES)
    output_csv_writer.writeheader()

    t0 = time.monotonic()
    failed = 0
    logger.info("Starting processing with %d workers", args.num_workers)
    pool = mp.Pool(processes=args.num_workers)
    try:
        for i, (clique_id, rows) in enumerate(
            pool.imap_unordered(_process_clique_worker, tasks, chunksize=1),
            start=1,
        ):
            if not rows:
                failed += 1
            else:
                output_csv_writer.writerows(rows)
                output_csv_file.flush()
                logger.info("%s processed", clique_id)
            if i % 100 == 0 or i == len(tasks):
                logger.info(
                    "Processed %d cliques out of %d",
                    i,
                    len(tasks),
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user; terminating workers.")
        pool.terminate()
        pool.join()
        sys.exit(0)
    else:
        pool.close()
        pool.join()
    finally:
        output_csv_file.close()

    if failed:
        logger.warning("%d cliques failed to process", failed)

    logger.info("Total time %s", s_to_hms(time.monotonic() - t0))
    logger.info("Done!")
