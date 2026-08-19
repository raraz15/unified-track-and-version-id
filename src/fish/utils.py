from pathlib import Path
from typing import Optional, Union
import math

import yaml

from .data import DevDataModule


def print_config(config: dict) -> None:
    print("\n\033[36mConfiguration:\033[0m")
    print(
        "\033[36m" + yaml.dump(config, indent=4, width=120, sort_keys=False) + "\033[0m"
    )


def load_config(config_path: Path | str, verbose: bool = True) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "cfg" in config:
        #  Unwrap so that we can pass a generated hparams.yaml to load_config as a config file
        config = config["cfg"]
    if verbose:
        print_config(config)
    return config


def get_version(ckpt_path: Union[str, Path]) -> Optional[Union[int, str]]:
    ckpt_parts = Path(ckpt_path).parts
    log_version = None
    for part in reversed(ckpt_parts):
        if part.startswith("version_"):
            suffix = part[len("version_") :]
            if suffix.isdigit():
                log_version = int(suffix)
            elif suffix:
                log_version = suffix
            break
    return log_version


def build_datamodule(cfg: dict, verbose: bool = True) -> DevDataModule:
    """Build the development datamodule."""

    train_cfg = cfg["train_dataloader"]
    train_cfg["context_duration"] = cfg["audio"]["context_duration"]

    if "augmentations" in cfg and "cqt_domain" in cfg["augmentations"]:
        aug_cfg = cfg["augmentations"]["cqt_domain"]
        if "time_stretching" in aug_cfg and aug_cfg["time_stretching"] is not None:
            # Calculate how much additional context is needed for time-stretching
            rmin = float(aug_cfg["time_stretching"]["rmin"])
            assert 1 >= rmin > 0.0, "rmin must be in (0.0, 1.0]"
            extra = train_cfg["context_duration"] * ((1 / rmin) - 1.0)
            # Round up to the nearest 0.1 second
            extra = math.ceil(extra * 10) / 10.0
            train_cfg["additional_context_duration"] = extra

    cliques_json_path = cfg["validation"]["cliques_json_path"]

    cfg_val_retr = cfg["validation"]["retrieval"]
    # NOTE: need to round because of floating point precision issues
    hop_duration = round(
        train_cfg["context_duration"] * (1 - cfg_val_retr["overlap_ratio"]),
        1,
    )

    clean_cfg = cfg_val_retr.get("clean_dataloader")
    if clean_cfg is not None:
        clean_cfg["hop_duration"] = hop_duration

    manip_cfg = cfg_val_retr.get("manipulated_and_degraded_dataloader")
    if manip_cfg is not None:
        manip_cfg["hop_duration"] = hop_duration

    cfg_val_ti = cfg["validation"].get("track_id")
    ti_clean_cfg = cfg_val_ti.get("clean_dataloader") if cfg_val_ti else None
    ti_manip_cfg = (
        cfg_val_ti.get("manipulated_and_degraded_dataloader") if cfg_val_ti else None
    )

    datamodule = DevDataModule(
        sample_rate=cfg["audio"]["sample_rate"],
        train_cfg=train_cfg,
        val_cliques_json_path=cliques_json_path,
        val_retrieval_clean_cfg=clean_cfg,
        val_retrieval_manip_cfg=manip_cfg,
        val_track_id_clean_cfg=ti_clean_cfg,
        val_track_id_manip_cfg=ti_manip_cfg,
        val_train_like_cfg=cfg["validation"]["alignment"],
        verbose=verbose,
    )
    return datamodule
