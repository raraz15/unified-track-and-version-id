import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.audio import load_audio, calculate_frame_rms_dBFS, SAMPLE_RATE


def _process_one(
    audio_path: Path,
    output_path: Path,
    frame_duration: float,
    threshold: float,
    ratio: float,
):

    assert threshold <= 0.0, "Threshold should be a non-positive dBFS value."
    assert 0.0 <= ratio <= 1.0, "Ratio should be in [0, 1]."

    audio = load_audio(audio_path, sample_rate=SAMPLE_RATE, n_channels=1)  # (1, T)
    if audio is None:
        print(f"Failed to load audio: {audio_path}")
        return

    frame_length = int(frame_duration * SAMPLE_RATE)

    # (1, N_frames)
    rms_dbfs = calculate_frame_rms_dBFS(audio, frame_length=frame_length)

    percentage = (rms_dbfs <= threshold).float().mean(dim=1).item()
    if percentage >= ratio:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{percentage:.2f} of frames below {threshold} dBFS\n")


def _process_task(task):

    audio_path, output_path, frame_duration, threshold, ratio = task
    _process_one(
        audio_path,
        output_path,
        frame_duration=frame_duration,
        threshold=threshold,
        ratio=ratio,
    )


def main(
    audio: Path,
    output_dir: Path,
    frame_duration: float,
    threshold: float,
    ratio: float,
    workers: Optional[int] = None,
):

    if audio.is_file():
        with audio.open("r", encoding="utf-8") as f:
            audio_paths = [
                Path(line.strip()) for line in f if line.strip() if line.strip()
            ]
            # Find the common directory for relative paths
            audio = Path(os.path.commonpath([str(p) for p in audio_paths]))
            print(f"Using common audio root directory: {audio}")
        print(f"Found {len(audio_paths):,} .wav files to analyze from list.")
    elif audio.is_dir():
        audio_paths = list(audio.rglob("*.wav"))
    else:
        raise ValueError(f"Invalid audio path: {audio}")
    print(f"Found {len(audio_paths):,} .wav files to analyze.")

    print("Preparing output paths...")
    tasks = []
    for audio_path in audio_paths:
        output_path = (output_dir / audio_path.relative_to(audio)).with_suffix(".txt")
        if not output_path.exists():
            tasks.append(
                (
                    audio_path,
                    output_path,
                    frame_duration,
                    threshold,
                    ratio,
                )
            )
    total = len(tasks)
    if total == 0:
        print("Nothing to do. All .wav files already processed.")
        return
    print(f"Found {total:,} .wav files to process.")

    print(f"Using frame duration of {frame_duration:.3f} s.")
    print(f"Marking files with >= {ratio:.2f} of frames below {threshold:.1f} dBFS.")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, _ in enumerate(executor.map(_process_task, tasks), start=1):
            if (i % 1000 == 0) or (i == total):
                print(f"Processed {i:,}/{total:,} files...")

    print(f"Done. Marked silent .wav files under: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Root directory containing .wav files or a text file listing .wav files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the silent-track marker .txt files.",
    )
    parser.add_argument("--frame-duration", type=float, default=0.100, help="s")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-50.0,
        help="dBFS threshold (<=0) to consider a frame silent.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.4,
        help="Minimum fraction of frames below threshold to mark a file as silent.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Default uses all.",
    )
    args = parser.parse_args()

    try:
        main(
            args.audio,
            args.output_dir,
            frame_duration=args.frame_duration,
            threshold=args.threshold,
            ratio=args.ratio,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
