"""Training script for Fish model using PyTorch Lightning and Weights & Biases logging."""

import os
import argparse
from pathlib import Path
from typing import Optional, Union

import torch
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from src.fish.utils import load_config, build_datamodule, get_version
from src.fish.model import FishModule
from src.common.utils import is_global_zero, SEED

# For TQDMProgressBar, adjust as needed for SLURM & Multi-GPU training
REFRESH_RATE = 50


def main(
    config_path: Union[Path, str],
    disable_wandb: bool = False,
    ckpt_path: Optional[Union[Path, str]] = None,
    log_version: Optional[Union[int, str]] = None,
    limit_train_batches: Optional[bool] = False,
    limit_val_batches: Optional[bool] = False,
    note_wandb: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> None:

    verbose = is_global_zero()

    cfg = load_config(config_path, verbose=verbose)

    if num_workers is not None:
        cfg["train_dataloader"]["num_workers"] = num_workers
    wandb_params = cfg.get("wandb", None)

    loggers, callbacks = [], []

    if log_version is None and ckpt_path is not None:
        log_version = get_version(ckpt_path)

    csv_logger = CSVLogger(cfg.get("log_root_dir", "logs/"), version=log_version)
    loggers.append(csv_logger)
    run_dir = Path(csv_logger.log_dir)
    if verbose:
        print(f"Run directory: {run_dir}")

    if not disable_wandb and wandb_params:
        from lightning.pytorch.loggers import WandbLogger

        wandb_id_file = run_dir / "wandb_run_id.txt"
        if ckpt_path is not None and wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            wandb_params["id"] = run_id
            wandb_params["resume"] = "allow"

        if note_wandb is not None:
            wandb_params["notes"] = note_wandb

        wandb_logger = WandbLogger(**wandb_params)
        loggers.append(wandb_logger)

        if verbose:
            run_id = wandb_logger.experiment.id  # str
            run_dir.mkdir(parents=True, exist_ok=True)
            wandb_id_file.write_text(run_id)

    # Always keep the latest checkpoint (no metric needed)
    callbacks.append(
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            save_last=True,
            save_on_train_epoch_end=True,
        )
    )
    # Save the best checkpoint based on validation metric
    callbacks.append(
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            monitor="val/comp_metric",
            mode="max",
        )
    )
    callbacks.append(TQDMProgressBar(refresh_rate=REFRESH_RATE))

    trainer = Trainer(
        logger=loggers,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        **cfg["training"]["parameters"],
    )

    datamodule = build_datamodule(cfg, verbose=verbose)

    module = FishModule(cfg)

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the YAML config file for training.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="""You can provide a path to a ckpt file to resume training.""",
    )
    parser.add_argument(
        "--log-version",
        type=str,
        default=None,
        help="Optional logger version. Use to resume into an existing log dir.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="You can limit the number of batches (int) for debugging.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="You can limit the number of batches (int) for debugging.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Wandb for logging."
    )
    parser.add_argument(
        "--note-wandb",
        type=str,
        default=None,
        help="Optional note to log to wandb.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override config train_dataloader num_workers.",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    seed_everything(SEED + rank, workers=True, verbose=True)

    torch.set_float32_matmul_precision("medium")

    main(
        args.train_config,
        disable_wandb=args.no_wandb,
        ckpt_path=args.ckpt,
        log_version=(
            int(args.log_version)
            if args.log_version is not None and args.log_version.isdigit()
            else args.log_version
        ),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        note_wandb=args.note_wandb,
        num_workers=args.num_workers,
    )
