import json
import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from lightning import Fabric
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure the repository root is on sys.path so local imports work when running the script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.utils import s_to_hms, get_git_commit
from src.common.audio import load_audio

# NOTE: this requires the original CLEWS library
clews_dir = "/home/raraz/version_identification/clews/"
if clews_dir not in sys.path:
    sys.path.append(clews_dir)

from models.clews import Model
from utils import pytorch_utils
from lib import tensor_ops as tops

MIN_DUR = 10.0  # seconds


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


def load_version_audio(
    audio_path: str | Path,
    sample_rate: int,
    min_dur: Optional[float] = None,
):
    """audio: 1,T"""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.warning("%s not found; skipping", audio_path)
        return None
    if min_dur is not None:
        min_length = int(min_dur * sample_rate)
        audio_len = torchaudio.info(str(audio_path)).num_frames
        if audio_len < min_length:
            logger.warning("%s too short %d; skipping", audio_path, audio_len)
            return None
    audio = load_audio(str(audio_path), sample_rate=sample_rate, n_channels=1)
    if audio is None:
        logger.warning("%s could not load", audio_path)
        return None
    if audio.numel() == 0:
        logger.warning("%s is empty; skipping", audio_path)
        return None
    return audio


class AudioDataset(Dataset):
    def __init__(
        self,
        io_pairs: list[tuple[str | Path, str | Path]],
        sr: int,
        min_dur: Optional[float] = None,
    ):
        """
        io_pairs: list of (input_filename, output_path) tuples
        sr: sample rate (e.g. 16000)
        """

        self.io_pairs = io_pairs
        self.sr = sr
        self.min_dur = min_dur

    def __len__(self):
        return len(self.io_pairs)

    def __getitem__(self, idx):

        path_audio, path_embedding = self.io_pairs[idx]
        x = load_version_audio(path_audio, sample_rate=self.sr, min_dur=self.min_dur)
        if x is None:
            return None

        return path_audio, path_embedding, x

    @staticmethod
    def collate_fn(batch):
        """Drop failed loads; return list of valid samples or None when empty."""
        valid = [sample for sample in batch if sample is not None]
        return valid if len(valid) > 0 else None


def embed(
    model,
    x,
    shingle_dur: float,
    shingle_hop_dur: float,
    fp_16: bool = True,
) -> torch.Tensor:
    """NOTE: This function does not enforce length to audio segments before cqt extraction."""

    assert x.ndim == 2, f"{x.shape}"

    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    shingle_len = int(shingle_dur * model.sr)
    shingle_hop_len = int(shingle_hop_dur * model.sr)

    with torch.inference_mode():
        # NOTE: I over-wrote the .forward of clews this way after various experiments.
        # You should stick to the official code of CLEWS. So much is going on here...
        x = tops.get_frames(x, shingle_len, shingle_hop_len, pad_end=True)
        x = x.squeeze(0)  # (N,L)
        x = model.cqt(x).unsqueeze(0)  # (1, N, F, T)
        x, _ = model.embed(x)  # (1, N, D)
        x = x.squeeze(0)  # (N, D)
        assert x.isfinite().all(), "Non-finite values in embedding!"
        if fp_16:
            x = x.half()
            assert (
                x.isfinite().all()
            ), "Non-finite values in embedding after converting to float16!"

    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract CLEWS embeddings from music files."
    )
    parser.add_argument(
        "music_dir",
        type=Path,
        help="Path to the directory containing the music files.",
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="""Path to the output directory to save the embeddings and logs.""",
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
        "--num-workers",
        default=6,
        type=int,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        type=int,
        help="Number of devices to use. Defaults to all visible GPUs if available.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Save embeddings in float16 precision.",
    )
    args = parser.parse_args()

    logger.info("Arguments: %s", vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    params["run_date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params["git_commit"] = get_git_commit()
    with open(args.output_dir / "parameters-clews-embeddings.json", "w") as out_f:
        json.dump(params, out_f, indent=4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(False)

    num_visible_gpus = torch.cuda.device_count()
    if args.devices is None:
        devices = "auto" if num_visible_gpus > 0 else 1
    else:
        devices = args.devices
    if devices == "auto":
        use_ddp = num_visible_gpus > 1
    else:
        use_ddp = devices > 1

    fabric = Fabric(
        accelerator="auto",
        devices=devices,
        precision="32",  # CLEWS was trained with FP32.
        strategy="ddp" if use_ddp else "auto",
    )
    fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size

    # Code from CLEWS repo to load the model
    logger.info("Loading configuration and checkpoint...")
    conf_path = args.checkpoint_path.parent / "configuration.yaml"
    conf = OmegaConf.load(conf_path)

    logger.info("Initializing model…")
    with fabric.init_module():
        model = Model(conf.model, sr=conf.data.samplerate)
    model = fabric.setup(model)
    model.mark_forward_method("embed")

    logger.info("Loading checkpoint...")
    state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
    fabric.load(args.checkpoint_path, state)
    model, _, _, conf, _, _, _ = pytorch_utils.set_state(state)
    model.eval()

    # Find the audio files to process
    if args.music_dir.is_dir():
        audio_paths = list(sorted(args.music_dir.rglob("*.wav")))
        base_dir = args.music_dir
    else:
        with open(args.music_dir) as f:
            audio_paths = [
                Path(line)
                for line in (l.strip() for l in f)
                if line.endswith(".wav") and Path(line).exists()
            ]
        base_dir = Path(os.path.commonpath(audio_paths))
    if fabric.is_global_zero:
        logger.info("%d audio files found.", len(audio_paths))

    # Filter out already processed files
    output_paths = [
        args.output_dir / p.relative_to(base_dir).with_suffix(".npy")
        for p in audio_paths
    ]
    io_pairs = [
        (p, out_p) for p, out_p in zip(audio_paths, output_paths) if not out_p.exists()
    ]
    if fabric.is_global_zero:
        logger.info("%d audio files to process.", len(io_pairs))

    dataset = AudioDataset(io_pairs, sr=model.sr, min_dur=MIN_DUR)
    sampler = None
    if world_size > 1:
        logger.info("Using DistributedSampler across %d ranks.", world_size)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
    )

    logger.info(
        "Rank %d assigned %d audio files (world size %d).",
        rank,
        len(loader),
        world_size,
    )
    t0, t_total = time.monotonic(), 0
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        audio_path, embedding_path, audio = batch[0]  # NOTE: batch size is 1
        try:
            z = embed(
                model,
                audio,
                shingle_dur=args.shingle_dur,
                shingle_hop_dur=args.shingle_hop_dur,
                fp_16=args.fp16,
            )
            z = z.cpu().numpy()
            embedding_path = Path(embedding_path)
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embedding_path, z)
        except KeyboardInterrupt:
            logger.info("Interrupted by user; exiting.")
            sys.exit(0)
        except Exception:
            logger.exception("Error processing audio file %s", audio_path)
        if (i + 1) == 1 or (i + 1) % 100 == 0 or i == len(loader) - 1:
            elapsed = time.monotonic() - t0
            logger.info("Step [%d/%d] took %s", i + 1, len(loader), s_to_hms(elapsed))
            t_total += elapsed
            t0 = time.monotonic()
    logger.info("Rank %d complete; total time %s", rank, s_to_hms(t_total))
    logger.info("Done!")
