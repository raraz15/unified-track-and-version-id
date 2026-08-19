"""Inference script for Fish model using PyTorch Lightning.

Loads a pre-trained checkpoint and runs predict() on an audio dataset,
writing embeddings to the specified output directory. The configuration is
always read from the checkpoint itself, so the architecture and the audio
front-end can never drift from the weights. The parameters that are free at
inference time (segment duration, overlap ratio) are exposed as arguments.
Supports multi-GPU via Lightning's Trainer and accepts truncation of long
audio files."""

import argparse
from pathlib import Path
from typing import Union, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar

from src.fish.utils import print_config
from src.fish.model import FishModule
from src.fish.data import InferenceDataset
from src.common.utils import set_seed, is_global_zero

# For TQDMProgressBar, adjust as needed for SLURM & Multi-GPU training
REFRESH_RATE = 50


def main(
    audio_path: Union[Path, str],
    ckpt_path: Union[Path, str],
    output_path: Union[Path, str],
    num_workers: int,
    batch_size: int,
    max_dur: Optional[float],
    segment_duration: Optional[float] = None,
    overlap_ratio: Optional[float] = None,
    accelerator: str = "gpu",
    inference_mode: bool = True,
) -> None:

    verbose = is_global_zero()

    # The checkpoint stores the config it was created with, so loading the
    # module from it guarantees that the architecture and the audio front-end
    # always match the weights. There is deliberately no way to override it.
    module = FishModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    cfg = module.hparams["cfg"]
    if verbose:
        print(f"Using the configuration stored in: {ckpt_path}")
        print_config(cfg)

    module.set_inference_params(
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
    )
    if verbose:
        print(
            "Segmentation: "
            f"segment_duration={segment_duration or cfg['audio']['context_duration']}s "
            f"(default {cfg['audio']['context_duration']}s), "
            f"overlap_ratio={overlap_ratio if overlap_ratio is not None else cfg['validation']['retrieval']['overlap_ratio']} "
            f"(default {cfg['validation']['retrieval']['overlap_ratio']})"
        )

    callbacks = [TQDMProgressBar(refresh_rate=REFRESH_RATE)]

    trainer = Trainer(
        accelerator=accelerator,
        precision=cfg["training"]["parameters"]["precision"],
        strategy=cfg["training"]["parameters"]["strategy"],
        enable_checkpointing=False,
        logger=False,
        callbacks=callbacks,  # type: ignore[arg-type]
        inference_mode=inference_mode,
    )

    dataset = InferenceDataset(
        input_path=audio_path,
        output_dir=output_path,
        max_duration=max_dur,
        sample_rate=cfg["audio"]["sample_rate"],
        verbose=verbose,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=dataset.collate_fn,
    )

    # The weights are already loaded, no ckpt_path needed here
    trainer.predict(
        model=module,
        dataloaders=dataloader,
        return_predictions=False,
    )

    print("Inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to a directory of audio files or a single audio file.",
    )
    parser.add_argument(
        "ckpt",
        type=Path,
        help="""Path to the pre-trained model ckpt file.""",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output directory where embeddings will be saved.",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=None,
        help="""Duration (in seconds) of the segments the audio is split into
        before embedding. Defaults to the context duration the model was
        trained with, which is stored in the ckpt file.""",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="""Overlap between consecutive segments, in [0.0, 1.0). Defaults
        to the retrieval overlap ratio stored in the ckpt file.""",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--no-inference-mode",
        action="store_false",
        dest="inference_mode",
        help="Disable inference_mode and use no_grad instead.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator type for Lightning Trainer (e.g. gpu, cpu).",
    )
    parser.add_argument(
        "--max-dur",
        type=float,
        default=None,
        help="""Maximum duration (in seconds) of audio files to process. 
        Longer files will be truncated.""",
    )
    args = parser.parse_args()

    set_seed()

    torch.set_float32_matmul_precision("medium")

    main(
        args.audio,
        args.ckpt,
        args.output_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_dur=args.max_dur,
        segment_duration=args.segment_duration,
        overlap_ratio=args.overlap_ratio,
        accelerator=args.accelerator,
        inference_mode=args.inference_mode,
    )
