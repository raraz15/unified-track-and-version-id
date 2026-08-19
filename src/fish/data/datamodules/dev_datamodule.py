from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

from src.common.audio import SAMPLE_RATE

from ..datasets import (
    SegmentCliqueDataset,
    SimilarSegmentPairDataset,
    ValRetrievalDataset,
    ValTIDataset,
)


class WeightedBatchSampler(BatchSampler):
    """Samples batches of distinct indices, weighted by per-item weights.

    Each batch is drawn independently via weighted sampling without replacement,
    so the full probability distribution resets every step.

    Subclasses PyTorch's BatchSampler so Lightning can recognize it in DDP.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        batch_size: int,
        steps_per_epoch: int,
    ) -> None:
        # Skip BatchSampler.__init__ — we don't use its sampler/drop_last logic
        self.weights = weights.float()
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.sampler = None  # required attribute for BatchSampler API
        self.drop_last = False  # required attribute for BatchSampler API
        assert (
            len(self.weights) >= self.batch_size
        ), f"Dataset has {len(self.weights)} items but batch_size is {self.batch_size}"

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            indices = torch.multinomial(
                self.weights, self.batch_size, replacement=False
            )
            yield indices.tolist()

    def __len__(self) -> int:
        return self.steps_per_epoch


class DevDataModule(LightningDataModule):
    def __init__(
        self,
        train_cfg: dict,
        val_cliques_json_path: Optional[str] = None,
        val_retrieval_clean_cfg: Optional[dict] = None,
        val_retrieval_manip_cfg: Optional[dict] = None,
        val_train_like_cfg: Optional[dict] = None,
        val_track_id_clean_cfg: Optional[dict] = None,
        val_track_id_manip_cfg: Optional[dict] = None,
        sample_rate: int = SAMPLE_RATE,
        verbose: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.train_cfg = train_cfg
        self.val_cliques_json_path = (
            Path(val_cliques_json_path) if val_cliques_json_path else None
        )
        self.val_retrieval_clean_cfg = val_retrieval_clean_cfg
        self.val_retrieval_manip_cfg = val_retrieval_manip_cfg
        self.val_train_like_cfg = val_train_like_cfg
        self.val_track_id_clean_cfg = val_track_id_clean_cfg
        self.val_track_id_manip_cfg = val_track_id_manip_cfg
        self.verbose = verbose

        self.num_workers = self.train_cfg.get("num_workers", 6)
        self.prefetch_factor = self.train_cfg.get("prefetch_factor", 1)

        self.dataset_val_retrieval_clean = None
        self.dataset_val_retrieval_manip = None
        self.dataset_val_align = None
        self.dataset_val_track_id_clean = None
        self.dataset_val_track_id_manip = None

    def setup(self, stage: str):

        if stage in {"fit", None}:
            self.dataset_train = SegmentCliqueDataset(
                csv_path=Path(self.train_cfg["csv_path"]),
                audio_dir=Path(self.train_cfg["audio_dir"]),
                context_duration=self.train_cfg["context_duration"],
                segments_per_tuple=self.train_cfg["segments_per_tuple"],
                augmentation_dict=self.train_cfg["augmentations"],
                additional_context_duration=self.train_cfg[
                    "additional_context_duration"
                ],
                sample_rate=self.sample_rate,
                verbose=self.verbose,
                noise_dir=self.train_cfg.get("background_noise_dir"),
                rir_dir=self.train_cfg.get("room_impulse_response_dir"),
                mir_dir=self.train_cfg.get("microphone_impulse_response_dir"),
            )

        if stage in {"fit", "validate", None}:
            if self.val_retrieval_clean_cfg is not None:
                self.dataset_val_retrieval_clean = ValRetrievalDataset(
                    cliques_json_path=self.val_cliques_json_path,
                    audio_dir=Path(self.val_retrieval_clean_cfg["audio_dir"]),
                    max_dur=self.val_retrieval_clean_cfg["max_duration"],
                    sample_rate=self.sample_rate,
                    clique_ratio=self.val_retrieval_clean_cfg.get("clique_ratio"),
                    verbose=self.verbose,
                )

            if self.val_retrieval_manip_cfg is not None:
                self.dataset_val_retrieval_manip = ValRetrievalDataset(
                    cliques_json_path=self.val_cliques_json_path,
                    audio_dir=Path(self.val_retrieval_manip_cfg["audio_dir"]),
                    max_dur=self.val_retrieval_manip_cfg["max_duration"],
                    sample_rate=self.sample_rate,
                    clique_ratio=self.val_retrieval_manip_cfg.get("clique_ratio"),
                    verbose=self.verbose,
                )

            if self.val_track_id_clean_cfg is not None:
                self.dataset_val_track_id_clean = ValTIDataset(
                    cliques_json_path=self.val_cliques_json_path,
                    audio_dir=Path(self.val_track_id_clean_cfg["audio_dir"]),
                    sample_rate=self.sample_rate,
                    verbose=self.verbose,
                )

            if self.val_track_id_manip_cfg is not None:
                self.dataset_val_track_id_manip = ValTIDataset(
                    cliques_json_path=self.val_cliques_json_path,
                    audio_dir=Path(self.val_track_id_manip_cfg["audio_dir"]),
                    sample_rate=self.sample_rate,
                    verbose=self.verbose,
                )

            if self.val_train_like_cfg:
                self.dataset_val_align = SegmentCliqueDataset(
                    csv_path=Path(self.val_train_like_cfg["csv_path"]),
                    audio_dir=self.val_train_like_cfg["audio_dir"],
                    context_duration=self.train_cfg["context_duration"],
                    segments_per_tuple=self.train_cfg["segments_per_tuple"],
                    sample_rate=self.sample_rate,
                    mode="Validation Loss",
                    verbose=self.verbose,
                )

    def train_dataloader(self) -> DataLoader:
        sampler = WeightedBatchSampler(
            weights=self.dataset_train.get_version_count_weights(),
            batch_size=self.train_cfg["tuples_per_batch"],
            steps_per_epoch=self.train_cfg["steps_per_epoch"],
        )
        return DataLoader(
            self.dataset_train,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.dataset_train.collate_fn,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=self.prefetch_factor,
        )

    def _val_sampler(self, dataset):
        """Use DistributedSampler when in DDP, otherwise default (sequential)."""
        if torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=False)
        return None

    def val_dataloader(self) -> list[DataLoader]:
        dataloaders = []

        if self.dataset_val_retrieval_clean is not None:
            clean_loader = DataLoader(
                self.dataset_val_retrieval_clean,
                batch_size=self.val_retrieval_clean_cfg["batch_size"],  # type: ignore
                sampler=self._val_sampler(self.dataset_val_retrieval_clean),
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=self.dataset_val_retrieval_clean.collate_fn,
                pin_memory=False,
                persistent_workers=False,
            )
            dataloaders.append(clean_loader)

        if self.dataset_val_retrieval_manip is not None:
            manip_loader = DataLoader(
                self.dataset_val_retrieval_manip,
                batch_size=self.val_retrieval_manip_cfg["batch_size"],  # type: ignore
                sampler=self._val_sampler(self.dataset_val_retrieval_manip),
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=self.dataset_val_retrieval_manip.collate_fn,
                pin_memory=False,
                persistent_workers=False,
            )
            dataloaders.append(manip_loader)

        if self.dataset_val_track_id_clean is not None:
            track_id_clean_loader = DataLoader(
                self.dataset_val_track_id_clean,
                batch_size=self.val_track_id_clean_cfg["batch_size"],  # type: ignore
                sampler=self._val_sampler(self.dataset_val_track_id_clean),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=self.dataset_val_track_id_clean.collate_fn,
                pin_memory=False,
                persistent_workers=False,
            )
            dataloaders.append(track_id_clean_loader)

        if self.dataset_val_track_id_manip is not None:
            track_id_manip_loader = DataLoader(
                self.dataset_val_track_id_manip,
                batch_size=self.val_track_id_manip_cfg["batch_size"],  # type: ignore
                sampler=self._val_sampler(self.dataset_val_track_id_manip),
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=self.dataset_val_track_id_manip.collate_fn,
                pin_memory=False,
                persistent_workers=False,
            )
            dataloaders.append(track_id_manip_loader)

        if self.dataset_val_align is not None:
            val_sampler = WeightedBatchSampler(
                weights=self.dataset_val_align.get_version_count_weights(),
                batch_size=self.train_cfg["tuples_per_batch"],
                steps_per_epoch=self.val_train_like_cfg["steps_per_epoch"],  # type: ignore
            )
            alignment_loader = DataLoader(
                self.dataset_val_align,
                batch_sampler=val_sampler,
                num_workers=self.num_workers,
                collate_fn=self.dataset_val_align.collate_fn,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=self.prefetch_factor,
            )
            dataloaders.append(alignment_loader)

        return dataloaders
