import os
import sys
import time
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.audio import load_audio, write_audio, SAMPLE_RATE

N_CHANNELS = 1


def process_audio(input_path: Path, output_path: Path):

    if (
        "OPENAIR" in str(input_path)
        and (
            "st-margarets-church-ncem-5-piece-band-spatial-measurements"
            not in str(input_path)
        )
        and input_path.parent.name == "stereo"
    ):

        x = load_audio(
            file_path=input_path,
            sample_rate=SAMPLE_RATE,
            n_channels=2,  # return both channels
            start=0,
            length=None,
            pad=False,
        )

        if x is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write both channels separately
            output_path_left = (
                output_path.parent / f"{output_path.stem}-channel_L{output_path.suffix}"
            )
            write_audio(output_path_left, x[0:1, :], sample_rate=SAMPLE_RATE)

            output_path_right = (
                output_path.parent / f"{output_path.stem}-channel_R{output_path.suffix}"
            )
            write_audio(output_path_right, x[1:2, :], sample_rate=SAMPLE_RATE)
    else:

        x = load_audio(
            file_path=input_path,
            sample_rate=SAMPLE_RATE,
            n_channels=N_CHANNELS,
            start=0,
            length=None,
            pad=False,
        )

        if x is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_audio(output_path, x, sample_rate=SAMPLE_RATE)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess degradation files.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the input audio files",
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

    print(f"Finding the .wav files in {str(args.input_dir)}...")
    input_paths = list(sorted(args.input_dir.rglob("*.wav")))
    print(f"Found {len(input_paths):,} audio files.")

    # To preserve the directory structure
    common_root = os.path.commonpath(input_paths)
    print(f"Common root: {common_root}")
    path_pairs = []
    for input_path in input_paths:
        relative_path = input_path.relative_to(common_root)
        output_path = args.output_dir / relative_path
        if not output_path.exists():
            path_pairs.append((input_path, output_path))
    print(f"Found {len(path_pairs):,} audio files to process.")

    # Shard to partitions
    path_pairs = path_pairs[args.partition_idx :: args.num_partitions]
    print(f"Processing {len(path_pairs):,} audio files...")
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
