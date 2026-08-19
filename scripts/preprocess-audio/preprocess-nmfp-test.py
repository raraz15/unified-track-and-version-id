import os
import sys
import time
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.audio import load_audio, write_audio, SAMPLE_RATE

N_CHANNELS = 1


def process_audio(input_path: Path, output_path: Path):

    x = load_audio(
        input_path,
        sample_rate=SAMPLE_RATE,
        n_channels=N_CHANNELS,
        start=0,
        length=None,
        backend="ffmpeg",  # FMA dataset has mp3 encoded files
        pad=False,
    )  # (1,T)

    if x is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_audio(output_path, x, sample_rate=SAMPLE_RATE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess NMFP Test set.")
    parser.add_argument(
        "inputs",
        type=Path,
        help="""Line delimited text file pointing to the files to be processed. 
        Since we only use the test set of NMFP, these should point to
        the original files in the FMA dataset.""",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the preprocessed audio files",
    )
    parser.add_argument(
        "--partition-idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    assert args.num_partitions > 0, "Number of partitions must be greater than 0"
    assert (
        args.partition_idx <= args.num_partitions
    ), "Partition must be less than or equal to num_partitions"

    with open(args.inputs) as i_f:
        input_paths = list(sorted([Path(p.strip()) for p in i_f.readlines()]))
    print(f"Found {len(input_paths):,} audio files.")

    # To preserve the directory structure
    common_root = os.path.commonpath(input_paths)
    path_pairs = []
    for input_path in input_paths:
        relative_path = input_path.relative_to(common_root)
        output_path = args.output_dir / relative_path.with_suffix(".wav")
        if not output_path.exists():
            path_pairs.append((input_path, output_path))
    print(f"Found {len(path_pairs):,} audio files to process.")

    # Shard to partitions
    path_pairs = path_pairs[args.partition_idx :: args.num_partitions]
    print(
        f"[{args.partition_idx}/{args.num_partitions}] Processing {len(path_pairs):,} audio files..."
    )
    if not path_pairs:
        print("No audio files to process.")
        sys.exit(0)

    t0, t_total = time.monotonic(), 0
    for i, (input_path, output_path) in enumerate(path_pairs):
        process_audio(input_path, output_path)
        if (i + 1) == 1 or (i + 1) % 100 == 0 or i == len(path_pairs) - 1:
            elapsed = time.monotonic() - t0
            print(
                f'Processed {i + 1:,} files [{time.strftime("%H:%M:%S", time.gmtime(elapsed))}].'
            )
            t_total += elapsed
            t0 = time.monotonic()
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(t_total))}")
    print("Done!")
