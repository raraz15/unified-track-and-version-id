import sys
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.optim as optim
from lightning.pytorch import LightningModule
from nnAudio import features

from src.common.audio import segmentation
from src.common.utils import is_global_zero
from src.fish.data.augmentations import AugmenterTime, AugmenterCQT
from src.fish.validation.retrieval_versions import retrieve_and_evaluate
from src.fish.validation.retrieval_tracks import identify_and_evaluate
from src.fish.validation.metrics import composite_metrics
from src.fish.model.cqt_process import CQTProcess
from src.fish.model import loss, nets
from src.fish.model.nets.clews import GeMPool, MyIBNResBlock

_NETWORKS = {"CLEWS": nets.CLEWS, "RESCQTNET": nets.ResCQTNet50}
_LOSSES = {
    "SUPCON": loss.SupConLoss,
    "INFONCE": loss.InfoNCELoss,
    "TRIPLET": loss.TripletLoss,
    "AU": loss.AULoss,
    "DCL": loss.DCL,
    "SUPDCL": loss.SupDCLLoss,
    "CONTRASTIVE": loss.ContrastiveLoss,
}
_OPTIMIZERS = {"adamw": optim.AdamW, "adam": optim.Adam, "sgd": optim.SGD}
_SCHEDULERS = {
    "reducelronplateau": optim.lr_scheduler.ReduceLROnPlateau,
    "cosineannealinglr": optim.lr_scheduler.CosineAnnealingLR,
    "cosineannealingwarmrestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "steplr": optim.lr_scheduler.StepLR,
    "multisteplr": optim.lr_scheduler.MultiStepLR,
    "exponentiallr": optim.lr_scheduler.ExponentialLR,
    "linearlr": optim.lr_scheduler.LinearLR,
}


class FishModule(LightningModule):
    def __init__(
        self,
        cfg: Dict[str, Any],
    ):
        super().__init__()

        # Stores cfg in self.hparams["cfg"] and in the checkpoints
        self.save_hyperparameters()

        # Inference-time overrides for extract_embeddings(). Lightning fixes the
        # predict_step() signature, so predict-only knobs are carried here.
        # Set via set_inference_params(), empty means "as configured".
        self._infer_kwargs: Dict[str, Any] = {}

        # Audio parameters
        audio_cfg = cfg["audio"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.context_duration = float(audio_cfg["context_duration"])
        self.context_length = int(self.context_duration * self.sample_rate)

        # Compilation flag
        network_cfg = dict(cfg["network"])
        self.compile_model = network_cfg.pop("compile", False)
        self._model_is_compiled = False

        # Initialize the network
        network_name = network_cfg.pop("name").upper()
        if is_global_zero():
            print(f"Network: {network_name}")
        if network_name not in _NETWORKS:
            raise ValueError(f"Unsupported network: {network_name}")
        self.model = _NETWORKS[network_name](**network_cfg)

        # Create the time-domain augmenter
        self.augmenter_time = None
        cfg_aug_time = cfg["augmentations"]["time_domain"]
        if cfg_aug_time:
            self.augmenter_time = AugmenterTime(dict(cfg_aug_time), eps=self.model.eps)

        # Create the input feature extractor
        feat_cfg = dict(cfg["input_feature"])
        self.feat_hop_dur = float(feat_cfg["hop_duration"])
        nbinsoct = int(feat_cfg["bins_per_octave"])
        self.feature_extractor = features.CQT1992v2(
            sr=self.sample_rate,
            fmin=float(feat_cfg["fmin"]),
            hop_length=int(self.feat_hop_dur * self.sample_rate),
            n_bins=int(feat_cfg["octaves"]) * nbinsoct,
            bins_per_octave=nbinsoct,
            verbose=False,
        )
        # NOTE: this holds for certain context_durations and hop durations
        # We use this so that the CQT augmentation can crop if needed.
        self.context_length_cqt = int(self.context_duration / self.feat_hop_dur) + 1
        self.n_cqt_bins = int(feat_cfg["octaves_use"]) * nbinsoct

        # To save time during inference
        if int(feat_cfg["octaves"]) * nbinsoct != self.n_cqt_bins:
            self.feature_extractor_inference = features.CQT1992v2(
                sr=self.sample_rate,
                fmin=float(feat_cfg["fmin"]),
                hop_length=int(self.feat_hop_dur * self.sample_rate),
                n_bins=self.n_cqt_bins,
                bins_per_octave=nbinsoct,
                verbose=False,
            )
        else:
            self.feature_extractor_inference = self.feature_extractor

        # Create the cqt-domain augmenter
        self.augmenter_cqt = None
        cfg_aug_cqt = cfg["augmentations"]["cqt_domain"]
        if cfg_aug_cqt:
            self.augmenter_cqt = AugmenterCQT(
                dict(cfg_aug_cqt),
                context_length_cqt=self.context_length_cqt,
                n_bins_cqt=self.n_cqt_bins,
            )

        # Feature processor (CQT to model input)
        self.feature_processor = CQTProcess(
            **dict(cfg["feature_process"]), eps=self.model.eps
        )

        # Loss function
        loss_cfg = dict(cfg["training"]["loss"])
        loss_name = loss_cfg.pop("name").upper()
        if is_global_zero():
            print(f"Loss: {loss_name}")
        if loss_name not in _LOSSES:
            raise ValueError(f"Unsupported loss: {loss_name}")
        self.loss_fn = _LOSSES[loss_name](**loss_cfg)
        self.loss_name = loss_name.lower()

        # VI Retrieval parameters for validation
        retrieval_dict = cfg["validation"]["retrieval"]
        self.val_batch_size = retrieval_dict["batch_size"]
        self.val_sim_search = retrieval_dict["similarity_search"]
        self.val_top_N = retrieval_dict["evaluation"]["top-N"]
        self.ckpt_metrics = retrieval_dict["evaluation"]["ckpt_metrics"]
        self.retrieval_overlap_ratio = float(retrieval_dict["overlap_ratio"])
        self.monitor = "val/comp_metric"
        self.comp_metric = torch.tensor(0.0, device=self.device)  # TODO

        # TI Retrieval parameters for validation
        ti_dict = cfg["validation"]["track_id"]
        self.val_ti_batch_size = ti_dict["batch_size"]

        self.log_kwargs = {"add_dataloader_idx": False, "on_step": True}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects a processed CQT input tensor of shape (B, 1, F, L) and returns the embeddings."""

        return self.model(x)

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            if not self._model_is_compiled:
                self.model = torch.compile(self.model)
                self._model_is_compiled = True

    def configure_optimizers(self):  # type: ignore
        train_cfg = dict(self.hparams["cfg"]["training"])

        opt_cfg = dict(train_cfg["optimizer"])
        opt_name = opt_cfg.pop("name").lower()
        if opt_name not in _OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        opt_class = _OPTIMIZERS[opt_name]
        if is_global_zero():
            print(f"Optimizer: {opt_class.__name__}")

        # Collect modules whose parameters should have no weight decay
        weight_decay_value = float(opt_cfg.pop("weight_decay"))
        if weight_decay_value == 0.0:
            opt = opt_class(self.parameters(), **opt_cfg)
        else:
            # Build a dict mapping parameters to their parent modules
            param_to_module = {}
            for module_name, module in self.named_modules():
                for param in module.parameters(recurse=False):
                    param_to_module[param] = module
            # Split parameters based on their parent module type
            no_decay = []
            with_decay = []
            for param in self.parameters():
                if not param.requires_grad:
                    continue

                parent_module = param_to_module.get(param)
                if parent_module is not None and (
                    isinstance(
                        parent_module,
                        (
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm1d,
                            torch.nn.InstanceNorm2d,
                            torch.nn.InstanceNorm1d,
                            GeMPool,
                            MyIBNResBlock,
                            loss.BaseLoss,
                            CQTProcess,
                        ),
                    )
                ):
                    no_decay.append(param)
                else:
                    with_decay.append(param)
            if is_global_zero():
                print(f"Parameters with weight decay: {len(with_decay)}")
                print(f"Parameters without weight decay: {len(no_decay)}")
            param_groups = [
                {"params": with_decay, "weight_decay": weight_decay_value},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            opt = opt_class(param_groups, **opt_cfg)

        if (
            "scheduler" not in train_cfg
            or train_cfg["scheduler"] is None
            or train_cfg["scheduler"] == dict()
        ):
            return opt
        sched_cfg = dict(train_cfg["scheduler"])
        sched_name = sched_cfg.pop("name", None).lower()
        if sched_name not in _SCHEDULERS:
            raise ValueError(f"Unsupported scheduler: {sched_name}")
        scheduler_class = _SCHEDULERS[sched_name]
        if is_global_zero():
            print(f"Scheduler: {scheduler_class.__name__}")

        if sched_name == "reducelronplateau":
            scheduler = scheduler_class(opt, mode="max", **sched_cfg)
            lr_scheduler_cfg = {
                "scheduler": scheduler,
                "monitor": self.monitor,
                "frequency": self.trainer.check_val_every_n_epoch,
                "strict": False,  # do not crash if monitor is missing for an epoch
            }
        else:
            interval = sched_cfg.pop("interval")
            frequency = sched_cfg.pop("frequency")
            scheduler = scheduler_class(opt, **sched_cfg)
            lr_scheduler_cfg = {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
            }

        return {"optimizer": opt, "lr_scheduler": lr_scheduler_cfg}

    def training_step(self, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train/lr",
            lr,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
        return self._shared_step(batch, prefix="train", batch_idx=batch_idx)

    def on_train_epoch_start(self):

        gem_p = self.model.back_end[-1].p.detach().view(-1).cpu().mean().item()  # type: ignore
        self.log("train/gem_p", gem_p, sync_dist=True)

        proj = self.model.projection  # type: ignore
        if hasattr(proj, "proj"):  # CLEWS Head
            weight = proj.proj.weight  # type: ignore
        elif hasattr(proj, "weight"):  # nn.Linear
            weight = proj.weight
        elif isinstance(proj, torch.nn.Sequential):  # bn-linear
            weight = proj[-1].weight
        else:
            weight = None
        if weight is not None:
            w = weight.detach()  # type: ignore
            proj_frob_norm = torch.linalg.norm(w, "fro").cpu().item()
            self.log("train/proj_frob_norm", proj_frob_norm)

            # Per output-dimension (row) L2 norms
            row_norms = torch.linalg.norm(w, dim=1)  # (out_dim,)
            self.log("train/proj_row_norm_mean", row_norms.mean().cpu().item())
            self.log("train/proj_row_norm_std", row_norms.std().cpu().item())
            self.log("train/proj_row_norm_max", row_norms.max().cpu().item())
            self.log("train/proj_row_norm_min", row_norms.min().cpu().item())

            # Singular value analysis
            sv = torch.linalg.svdvals(w.float())  # (min(out,in),)
            self.log("train/proj_sv_max", sv[0].cpu().item())
            self.log("train/proj_sv_min", sv[-1].cpu().item())
            self.log(
                "train/proj_sv_ratio", (sv[0] / sv[-1].clamp(min=1e-8)).cpu().item()
            )
            # Effective rank: exp(entropy of normalized singular values)
            sv_norm = sv / sv.sum()
            eff_rank = torch.exp(-(sv_norm * sv_norm.clamp(min=1e-12).log()).sum())
            self.log("train/proj_effective_rank", eff_rank.cpu().item())

        if hasattr(self.feature_processor, "gain"):
            self.log("train/cqt_gain", self.feature_processor.gain.detach().item())
            self.log("train/cqt_bias", self.feature_processor.bias.detach().item())

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0:
            emb, c_ids, v_ids, sizes = self._db_emb, self._db_c_ids, self._db_v_ids, self._db_sizes  # fmt: skip
        elif dataloader_idx == 1:
            emb, c_ids, v_ids, sizes = self._q_emb, self._q_c_ids, self._q_v_ids, self._q_sizes  # fmt: skip
        elif dataloader_idx == 2:
            emb, c_ids, v_ids, sizes = self._ti_clean_emb, self._ti_clean_c_ids, self._ti_clean_v_ids, self._ti_clean_sizes  # fmt: skip
        elif dataloader_idx == 3:
            emb, c_ids, v_ids, sizes = self._ti_manip_emb, self._ti_manip_c_ids, self._ti_manip_v_ids, self._ti_manip_sizes  # fmt: skip
        else:
            # Validation loss loader
            return self._shared_step(batch, prefix="val", batch_idx=batch_idx)

        # Retrieval loaders: expect (audio, clique_id, version_id, lengths)
        audio_clips, clique_ids, version_ids, lengths = batch
        z_batch = self.extract_embeddings(audio_clips, true_lengths=lengths)
        for z, id_c, id_v in zip(z_batch, clique_ids, version_ids):
            # Move to CPU to avoid fragmenting GPU memory
            emb.append(z.cpu())
            c_ids.append(int(id_c.cpu()))
            v_ids.append(int(id_v.cpu()))
            sizes.append(int(z.shape[0]))

    def on_validation_epoch_start(self):
        self._db_emb, self._db_c_ids, self._db_v_ids, self._db_sizes = [], [], [], []
        self._q_emb, self._q_c_ids, self._q_v_ids, self._q_sizes = [], [], [], []
        (
            self._ti_clean_emb,
            self._ti_clean_c_ids,
            self._ti_clean_v_ids,
            self._ti_clean_sizes,
        ) = ([], [], [], [])
        (
            self._ti_manip_emb,
            self._ti_manip_c_ids,
            self._ti_manip_v_ids,
            self._ti_manip_sizes,
        ) = ([], [], [], [])

    def on_validation_epoch_end(self):

        device = self.device
        multi_gpu = self.trainer.world_size > 1

        # Concatenate on RAM to avoid VRAM fragmentation
        db_embs, db_c_ids, db_v_ids, db_sizes = self._collect_and_clear(
            self._db_emb, self._db_c_ids, self._db_v_ids, self._db_sizes
        )
        q_embs, q_c_ids, q_v_ids, q_sizes = self._collect_and_clear(
            self._q_emb, self._q_c_ids, self._q_v_ids, self._q_sizes
        )

        if self._ti_clean_emb:
            ti_clean_embs, _, ti_clean_track_ids, ti_clean_sizes = (
                self._collect_and_clear(
                    self._ti_clean_emb,
                    self._ti_clean_c_ids,
                    self._ti_clean_v_ids,
                    self._ti_clean_sizes,
                )
            )
        else:
            ti_clean_embs = torch.empty(0)
            ti_clean_track_ids = torch.empty(0, dtype=torch.long)
            ti_clean_sizes = torch.empty(0, dtype=torch.long)

        if self._ti_manip_emb:
            ti_manip_embs, _, ti_manip_track_ids, ti_manip_sizes = (
                self._collect_and_clear(
                    self._ti_manip_emb,
                    self._ti_manip_c_ids,
                    self._ti_manip_v_ids,
                    self._ti_manip_sizes,
                )
            )
        else:
            ti_manip_embs = torch.empty(0)
            ti_manip_track_ids = torch.empty(0, dtype=torch.long)
            ti_manip_sizes = torch.empty(0, dtype=torch.long)

        # Release CUDA cache from training before allocating gather tensors
        torch.cuda.empty_cache()

        # Move to device
        db_embs, db_c_ids, db_v_ids, db_sizes = [
            t.to(device) for t in (db_embs, db_c_ids, db_v_ids, db_sizes)
        ]
        q_embs, q_c_ids, q_v_ids, q_sizes = [
            t.to(device) for t in (q_embs, q_c_ids, q_v_ids, q_sizes)
        ]
        ti_clean_embs, ti_clean_track_ids, ti_clean_sizes = [
            t.to(device) for t in (ti_clean_embs, ti_clean_track_ids, ti_clean_sizes)
        ]
        ti_manip_embs, ti_manip_track_ids, ti_manip_sizes = [
            t.to(device) for t in (ti_manip_embs, ti_manip_track_ids, ti_manip_sizes)
        ]

        # Gather across ranks if multi-GPU
        if multi_gpu:
            db_embs, db_c_ids, db_v_ids, db_sizes = self._gather_val_embeddings(
                db_embs, db_c_ids, db_v_ids, db_sizes
            )
            q_embs, q_c_ids, q_v_ids, q_sizes = self._gather_val_embeddings(
                q_embs, q_c_ids, q_v_ids, q_sizes
            )

            # TODO a function
            ti_clean_embs = self.all_gather(ti_clean_embs).reshape(
                -1, ti_clean_embs.shape[-1]
            )
            ti_clean_track_ids = self.all_gather(ti_clean_track_ids).reshape(-1)
            ti_clean_sizes = self.all_gather(ti_clean_sizes).reshape(-1)

            ti_manip_embs = self.all_gather(ti_manip_embs).reshape(
                -1, ti_manip_embs.shape[-1]
            )
            ti_manip_track_ids = self.all_gather(ti_manip_track_ids).reshape(-1)
            ti_manip_sizes = self.all_gather(ti_manip_sizes).reshape(-1)

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        q_sizes = q_sizes.cpu().tolist()
        db_sizes = db_sizes.cpu().tolist()
        ti_clean_sizes = ti_clean_sizes.cpu().tolist()
        ti_manip_sizes = ti_manip_sizes.cpu().tolist()

        if not multi_gpu or is_global_zero():
            log_kw = dict(on_epoch=True, sync_dist=True, rank_zero_only=multi_gpu)

            print("VI Querying the clean tracks...", flush=True)
            clean_metrics = retrieve_and_evaluate(
                db_embs,
                db_c_ids,
                db_v_ids,
                db_sizes,
                db_embs,
                db_c_ids,
                db_v_ids,
                db_sizes,
                similarity_search=self.val_sim_search,
                batch_size=self.val_batch_size,
                k=self.val_top_N,
            )
            for k, v in clean_metrics.items():
                if "CI" not in k:  # Do not log the confidence values
                    self.log(f"val/{k}", v, **log_kw)  # type: ignore

            # Checkpoint follows clean metrics
            comp = composite_metrics(clean_metrics, self.ckpt_metrics)
            self.comp_metric.fill_(comp)

            print("VI Querying the manipulated and degraded tracks...", flush=True)
            manip_metrics = retrieve_and_evaluate(
                q_embs,
                q_c_ids,
                q_v_ids,
                q_sizes,
                db_embs,
                db_c_ids,
                db_v_ids,
                db_sizes,
                similarity_search=self.val_sim_search,
                batch_size=self.val_batch_size,
                k=self.val_top_N,
            )
            for k in ["M-AP", "M-JNAR"]:
                self.log(f"val/manip-deg-{k}", manip_metrics[k], **log_kw)  # type: ignore

            # Track identification evaluation — clean queries
            if ti_clean_sizes:
                print("TI Querying the clean chunks...", flush=True)
                ti_clean_metrics = identify_and_evaluate(
                    ti_clean_embs,
                    ti_clean_track_ids,
                    ti_clean_sizes,
                    db_embs,
                    db_v_ids,
                    db_sizes,
                    similarity_search=self.val_sim_search,
                    batch_size=self.val_ti_batch_size,
                )
                self.log("val/ti-clean-top1_hit_rate", ti_clean_metrics["top1_hit_rate"], **log_kw)  # type: ignore
                self.log("val/ti-clean-top10_hit_rate", ti_clean_metrics["top10_hit_rate"], **log_kw)  # type: ignore

            # Track identification evaluation — manipulated and degraded queries
            if ti_manip_sizes:
                print("TI Querying the manipulated and degraded chunks...", flush=True)
                ti_manip_metrics = identify_and_evaluate(
                    ti_manip_embs,
                    ti_manip_track_ids,
                    ti_manip_sizes,
                    db_embs,
                    db_v_ids,
                    db_sizes,
                    similarity_search=self.val_sim_search,
                    batch_size=self.val_ti_batch_size,
                )
                self.log("val/ti-manip-deg-top1_hit_rate", ti_manip_metrics["top1_hit_rate"], **log_kw)  # type: ignore
                self.log("val/ti-manip-deg-top10_hit_rate", ti_manip_metrics["top10_hit_rate"], **log_kw)  # type: ignore

        # Free memory
        del db_embs, db_c_ids, db_v_ids, db_sizes
        del q_embs, q_c_ids, q_v_ids, q_sizes
        del ti_clean_embs, ti_clean_track_ids, ti_clean_sizes
        del ti_manip_embs, ti_manip_track_ids, ti_manip_sizes
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if multi_gpu:
            self.comp_metric = self.trainer.strategy.broadcast(self.comp_metric, src=0)
        self.log(
            self.monitor,
            self.comp_metric,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=multi_gpu,
        )

    def set_inference_params(self, **kwargs) -> None:
        """Override extract_embeddings() parameters used during predict().

        Accepts any keyword of extract_embeddings() except audio/true_lengths,
        e.g. segment_duration and overlap_ratio. None values are dropped so the
        configured defaults survive. Only affects how audio is segmented and
        read out; weights, CQT front-end and feature processing are untouched."""

        allowed = {
            "segment_duration",
            "overlap_ratio",
            "layer",
            "normalize",
            "chunk_size",
            "output_half",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"Unknown inference parameters: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}."
            )
        self._infer_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def predict_step(self, batch, _) -> None:

        audio_clips, input_paths, output_paths, lengths = batch
        try:
            zs = self.extract_embeddings(
                audio_clips, true_lengths=lengths, **self._infer_kwargs
            )
            for z, path_out in zip(zs, output_paths):
                z = z.detach().cpu().numpy()
                path_out = Path(path_out)
                path_out.parent.mkdir(exist_ok=True, parents=True)
                np.save(path_out, z)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(
                f"[Rank {self.global_rank}] Extraction failed on {input_paths}: {repr(e)}",
                flush=True,
            )
            traceback.print_exc()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # When torch.compile() is active, state dict keys get "_orig_mod." inserted.
        # Strip it so saved checkpoints are always portable (loadable without compile).
        state_dict = checkpoint["state_dict"]
        if any("._orig_mod." in k for k in state_dict):
            checkpoint["state_dict"] = {
                k.replace("._orig_mod.", "."): v for k, v in state_dict.items()
            }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Compiled graphs are not saved in checkpoints, so force recompilation after restore.
        self._model_is_compiled = False
        state_dict = checkpoint["state_dict"]
        has_orig_mod = any("._orig_mod." in k for k in state_dict)
        model_is_compiled = any("._orig_mod." in k for k in self.state_dict())

        if has_orig_mod and not model_is_compiled:
            # Backwards compat: strip _orig_mod. from old checkpoints
            checkpoint["state_dict"] = {
                k.replace("._orig_mod.", "."): v for k, v in state_dict.items()
            }
        elif not has_orig_mod and model_is_compiled:
            # Checkpoint was saved without compile but model is currently compiled;
            # add _orig_mod. prefix so keys match the compiled model
            new_state_dict = {}
            for k, v in state_dict.items():
                # Keys are like "model.front_end.0.weight" -> "model._orig_mod.front_end.0.weight"
                if k.startswith("model."):
                    new_state_dict["model._orig_mod." + k[len("model.") :]] = v
                else:
                    new_state_dict[k] = v
            checkpoint["state_dict"] = new_state_dict

    @torch.no_grad()  # With torch.compile, no_grad did not slow down training
    def extract_embeddings(
        self,
        audio: torch.Tensor,  # (B, 1, T) or (1, T)
        true_lengths: Optional[list[int]] = None,  # (B,)
        segment_duration: Optional[float] = None,
        overlap_ratio: Optional[float] = None,
        layer: Optional[str] = "proj",  # TODO this should not have a default param?
        normalize: bool = False,
        chunk_size: int = 1024,
        output_half: bool = True,
    ) -> tuple[torch.Tensor, ...]:

        # The network is fully convolutional and GeM-pooled, so it accepts any
        # segment length. self.context_length stays what the model was trained
        # with and is only the default here.
        if segment_duration is None:
            segment_len = self.context_length  # as trained
        else:
            segment_len = int(segment_duration * self.sample_rate)
            # Below ~0.7 s the front-end convolutions (time kernel 3, stride 2,
            # no time padding, applied after the feature downsampling) leave no
            # time frames and the forward pass dies with a shape error.
            assert (
                segment_len >= self.sample_rate
            ), f"segment_duration must be >= 1.0 s, got {segment_duration}."

        if overlap_ratio is None:
            overlap_ratio = self.retrieval_overlap_ratio
        assert 0.0 <= overlap_ratio < 1.0, "overlap_ratio must be in [0.0, 1.0)."
        hop_len = int(segment_len * (1 - overlap_ratio))

        if audio.ndim == 2:
            assert audio.shape[0] == 1, "Input audio must have shape (1, T)."
            audio = audio.unsqueeze(0)  # (1, 1, T)
            true_lengths = [audio.shape[-1]]
        elif audio.ndim == 3:
            assert audio.shape[1] == 1, "Input audio must have shape (B, 1, T)."
            assert (
                true_lengths is not None
            ), "If you provide batched inputs, indicate original lengths!"
        else:
            raise ValueError("Audio must be be 2D or 3D (i.e., batch)!")

        # Remove the padding for each audio clip and segment
        segments = []
        for x, true_length in zip(audio, true_lengths):
            x = x[:, :true_length]  # (1, T_x)
            segments.append(
                segmentation(x, segment_len, hop_len, pad_mode="constant")
            )  # (N_x, 1, L)
        n_segments = [s.shape[0] for s in segments]
        segments = torch.cat(segments, dim=0)  # (N, 1, L)

        # We embed the segments in chunks to avoid OOM
        embeddings = []
        for i in range(0, segments.shape[0], chunk_size):
            chunk = segments[i : i + chunk_size]
            # NOTE: this computes CQTs for the overlapped regions twice but the model
            # only saw these kinds of CQTs during training (last frame)
            x = self._audio_to_input(chunk)  # (N_chunk, 1, F, Lf)
            x = self.model.embed(x, layer=layer, normalize=normalize)  # type: ignore
            embeddings.append(x)
        embeddings = torch.cat(embeddings, dim=0)  # Re-assemble (N, D)
        assert embeddings.isfinite().all(), "Non-finite values in embedding!"

        # Convert to half precision if needed
        if output_half:
            embeddings = embeddings.half()
            assert (
                embeddings.isfinite().all()
            ), "Non-finite values in embedding after converting to FP16!"

        # Split to corresponding tracks
        embeddings = torch.split(embeddings, n_segments, dim=0)

        return embeddings

    def _shared_step(self, batch: tuple, prefix: str, batch_idx: int) -> torch.Tensor:
        """Shared step for training and validation."""

        # x: (2*N,1,T) labels: (2*N,)
        # noise, rir, mir: (2*N,1,T) or None
        x, noise, rir, mir, labels = batch[:5]

        x = self._audio_to_input(x, noise=noise, rir=rir, mir=mir)

        x = self.forward(x)

        loss_dict = self.loss_fn(x, labels)

        # Log all loss components
        for k, v in loss_dict.items():
            kwargs = dict(on_epoch=False, rank_zero_only=True)
            if k == "loss":
                k = f"loss-{self.loss_name}"
                if prefix == "train":
                    kwargs["prog_bar"] = True
                else:
                    kwargs.update(on_epoch=True, batch_size=x.shape[0], sync_dist=True)  # type: ignore
            self.log(f"{prefix}/{k}", v, **kwargs, **self.log_kwargs)  # type: ignore

        return loss_dict["loss"]

    def _audio_to_input(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        rir: Optional[torch.Tensor] = None,
        mir: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, 1, T) returns (B, 1, F, T)."""

        assert x.ndim == 3, "Input audio must have shape (B, 1, T)"
        assert x.shape[1] == 1, f"Input audio must be mono. {x.shape}"

        with torch.no_grad():
            if self.training:
                # The order of operations is not realistic but its way faster to time stretch etc. on the CQT domain
                if self.augmenter_time is not None:
                    x = self.augmenter_time(x, noise=noise, rir=rir, mir=mir)
                # Oversized train CQT, then let the augmenter crop it back
                x = self.feature_extractor(x)  # (B, F, L')
                if self.augmenter_cqt is not None:
                    x = self.augmenter_cqt(x)
            else:
                x = self.feature_extractor_inference(x)  # (B, F, L')

        # Check if feature processor has learnable parameters to decide on no_grad context
        if any(p.requires_grad for p in self.feature_processor.parameters()):
            x = self.feature_processor(x)  # (B, F, L)
        else:
            with torch.no_grad():
                x = self.feature_processor(x)  # (B, F, L)

        return x.unsqueeze(1)  # (B, 1, F, L)

    def _collect_and_clear(self, emb_list, c_ids_list, v_ids_list, sizes_list):
        """Concatenate local validation embeddings on CPU and clear the buffers."""
        embs = torch.cat(emb_list, dim=0)
        c_ids = torch.tensor(c_ids_list, dtype=torch.long)
        v_ids = torch.tensor(v_ids_list, dtype=torch.long)
        sizes = torch.tensor(sizes_list, dtype=torch.long)
        emb_list.clear()
        c_ids_list.clear()
        v_ids_list.clear()
        sizes_list.clear()
        return embs, c_ids, v_ids, sizes

    def _gather_val_embeddings(self, local_embs, local_c_ids, local_v_ids, local_sizes):
        device = self.device
        # Determine how much padding is needed per rank to call all_gather
        seg_count_local = torch.tensor(
            local_embs.shape[0],
            device=device,
            dtype=torch.int64,
        )
        track_count_local = torch.tensor(
            local_c_ids.shape[0], device=device, dtype=torch.int64
        )
        seg_count_gathered = self.all_gather(seg_count_local)
        track_count_gathered = self.all_gather(track_count_local)
        max_seg_count = int(seg_count_gathered.amax().item())  # type: ignore
        max_track_count = int(track_count_gathered.amax().item())  # type: ignore
        # Padding to be applied locally
        local_pad_amount_seg = max_seg_count - int(seg_count_local.item())
        local_pad_amount_track = max_track_count - int(track_count_local.item())

        # Pad the local tensors to equal length
        local_embs = torch.nn.functional.pad(
            local_embs,
            (0, 0, 0, local_pad_amount_seg),
            value=0.0,
        )
        local_c_ids = torch.nn.functional.pad(
            local_c_ids,
            (0, local_pad_amount_track),
            value=-1,
        )
        local_v_ids = torch.nn.functional.pad(
            local_v_ids,
            (0, local_pad_amount_track),
            value=-1,
        )
        local_sizes = torch.nn.functional.pad(
            local_sizes,
            (0, local_pad_amount_track),
            value=0,
        )

        # Can gather now
        embs_padded = self.all_gather(local_embs)  # (Nworld, Ns, D)
        c_ids_padded = self.all_gather(local_c_ids)  # (Nworld, Nt)
        v_ids_padded = self.all_gather(local_v_ids)  # (Nworld, Nt)
        sizes_padded = self.all_gather(local_sizes)  # (Nworld, Nt)

        # Remove the paddings and concat
        embs, c_ids, v_ids, sizes = [], [], [], []
        for i in range(embs_padded.shape[0]):  # type: ignore
            s = int(seg_count_gathered[i].item())
            t = int(track_count_gathered[i].item())
            embs.append(embs_padded[i, :s])  # type: ignore
            c_ids.append(c_ids_padded[i, :t])  # type: ignore
            v_ids.append(v_ids_padded[i, :t])  # type: ignore
            sizes.append(sizes_padded[i, :t])  # type: ignore
        embs = torch.cat(embs)
        c_ids = torch.cat(c_ids)
        v_ids = torch.cat(v_ids)
        sizes = torch.cat(sizes)
        assert embs.shape[0] == int(sizes.sum().item())

        # Free memory
        del embs_padded, c_ids_padded, v_ids_padded, sizes_padded
        del seg_count_gathered, track_count_gathered

        return embs, c_ids, v_ids, sizes
