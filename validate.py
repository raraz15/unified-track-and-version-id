"""Validation script for Fish model using PyTorch Lightning.

The configuration is read from the checkpoint itself, so a config file is only
needed to override it."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar

from src.fish.utils import load_config, print_config, build_datamodule
from src.fish.model import FishModule
from src.common.utils import is_global_zero, SEED

# For TQDMProgressBar, adjust as needed for SLURM & Multi-GPU training
REFRESH_RATE = 50


def main(
    ckpt_path: Union[Path, str],
    config_path: Optional[Path] = None,
    limit_val_batches: Optional[int] = None,
    disable_alignment: bool = False,
    disable_retrieval: bool = False,
) -> None:

    verbose = is_global_zero()

    # The checkpoint stores the config it was created with, so loading the
    # module from it guarantees that the architecture and the audio front-end
    # match the weights. A config file is only needed to override them.
    if config_path is None:
        module = FishModule.load_from_checkpoint(ckpt_path, map_location="cpu")
        cfg = module.hparams["cfg"]
        if verbose:
            print(f"Using the configuration stored in: {ckpt_path}")
            print_config(cfg)
    else:
        cfg = load_config(config_path, verbose=verbose)
        module = FishModule.load_from_checkpoint(
            ckpt_path, map_location="cpu", cfg=cfg
        )

    log_dir = Path(ckpt_path).parent.parent / "validation"  # .../version_X/validation
    version = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    csv_logger = CSVLogger(
        save_dir=str(log_dir),
        name=".",
        version=version,
    )
    if verbose:
        print(f"Saving the logs to: {str(Path(csv_logger.log_dir).resolve())}")

    callbacks = [TQDMProgressBar(refresh_rate=REFRESH_RATE)]

    cfg["training"]["parameters"]["benchmark"] = False
    trainer = Trainer(
        logger=csv_logger,
        callbacks=callbacks,  # type: ignore[arg-type]
        enable_checkpointing=False,
        limit_val_batches=limit_val_batches,
        inference_mode=True,
        **cfg["training"]["parameters"],
    )

    if disable_alignment:
        if verbose:
            print("Disabling alignment loss calculation during validation.")
        cfg["validation"]["alignment"] = None

    if disable_retrieval:
        if verbose:
            print("Disabling retrieval loss calculation during validation.")
        cfg["validation"]["retrieval"] = None

    datamodule = build_datamodule(cfg, verbose=verbose)

    # The weights are already loaded, no ckpt_path needed here
    results = trainer.validate(model=module, datamodule=datamodule)

    if verbose:
        for result in results:
            print(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "ckpt",
        type=Path,
        help="""Path to the pre-trained model ckpt file.""",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="""Path to a YAML config file. By default the config stored in the
        ckpt file is used, provide this only to override it.""",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="You can limit the number of batches (int) for debugging.",
    )
    parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="Disable retrieval loss calculation during validation.",
    )
    parser.add_argument(
        "--disable-alignment",
        action="store_true",
        help="Disable alignment loss calculation during validation.",
    )
    args = parser.parse_args()

    # 27 is the license plate code of Gaziantep, gastronomical capital of Türkiye.
    seed_everything(SEED, workers=True, verbose=True)

    torch.set_float32_matmul_precision("medium")

    main(
        args.ckpt,
        config_path=args.config,
        limit_val_batches=args.limit_val_batches,
        disable_alignment=args.disable_alignment,
        disable_retrieval=args.disable_retrieval,
    )
