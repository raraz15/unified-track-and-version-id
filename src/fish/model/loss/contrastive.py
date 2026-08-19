import torch
import torch.nn.functional as F

from src.common.tensor_op import pairwise_distance_matrix
import src.fish.model.loss.triplet_mining as triplet_mining

from .base import BaseLoss


class ContrastiveLoss(BaseLoss):
    def __init__(
        self,
        margin: float = 0.3,
        positive_mining_mode: str = "hard",
        negative_mining_mode: str = "hard",
        squared_distance: bool = False,
        normalize_distance: bool = False,
        pos_weight: float = 1.0,
    ):

        super().__init__()

        assert pos_weight > 0, "pos_weight must be greater than 0."
        self.pos_weight = pos_weight

        assert margin >= 0, "margin must be greater or equal to 0."
        assert positive_mining_mode.lower() in [
            "all",
            "random",
            "easy",
            "hard",
        ], "Positive mining mode must be either 'all', 'random', 'easy', or 'hard'."
        if negative_mining_mode.lower() == "class-hard":
            raise NotImplementedError(
                "class-hard negative mining is not supported. Use 'all', 'semi-hard', or 'hard'."
            )
        assert negative_mining_mode.lower() in [
            "all",
            "semi-hard",
            "hard",
        ], "Negative mining mode must be either 'all', 'semi-hard', or 'hard'."

        self.margin = margin
        self.positive_mining_mode = positive_mining_mode.lower()
        self.negative_mining_mode = negative_mining_mode.lower()

        if squared_distance and normalize_distance:
            self.dist = "nsqeuc"
        elif squared_distance and not normalize_distance:
            self.dist = "sqeuc"
        elif not squared_distance and normalize_distance:
            self.dist = "neuc"
        else:
            self.dist = "euc"

    def forward(self, embeddings: torch.Tensor, clique_ids: torch.Tensor) -> dict:

        mask_pos, mask_neg = self._get_pos_neg_masks(clique_ids=clique_ids)
        distance_matrix = pairwise_distance_matrix(embeddings, mode=self.dist)

        loss_dict = self._get_loss_dict(distance_matrix, mask_pos, mask_neg)
        loss_dict["loss"] = loss_dict["loss"].mean()

        return loss_dict

    def _get_loss_dict(
        self,
        distance_matrix: torch.Tensor,
        mask_pos: torch.Tensor,
        mask_neg: torch.Tensor,
    ) -> dict:

        if self.positive_mining_mode == "all" and self.negative_mining_mode == "all":
            return self._batch_all(distance_matrix, mask_pos, mask_neg)
        return self._batch_hard(distance_matrix, mask_pos, mask_neg)

    def _mine_positives(
        self, distance_matrix: torch.Tensor, mask_pos: torch.Tensor
    ) -> torch.Tensor:

        if self.positive_mining_mode == "random":
            dist_AP, _ = triplet_mining.random_positive_sampling(
                distance_matrix, mask_pos
            )
        elif self.positive_mining_mode == "hard":
            dist_AP, _ = triplet_mining.hard_positive_mining(distance_matrix, mask_pos)
        else:
            dist_AP, _ = triplet_mining.easy_positive_mining(distance_matrix, mask_pos)
        return dist_AP

    def _build_loss_dict(
        self,
        dist_AP: torch.Tensor,
        dist_AN: torch.Tensor,
        mask_pos: torch.Tensor,
    ) -> dict:

        loss_pos = dist_AP.mean()

        # NOTE: in the second paper, they square the ReLU output. I square the distance instead.
        # Not squaring is also possible
        neg_violations = F.relu(self.margin - dist_AN)
        num_unsatisfied = int((neg_violations > 0).sum().item())
        loss_neg = neg_violations.mean()

        loss = self.pos_weight * loss_pos + loss_neg

        return {
            "loss": loss,
            "num_unsatisfied_negatives": num_unsatisfied,
            "max_class_size": mask_pos.sum(1).int().max().item() + 1,
            "dist_AP_mean": loss_pos.item(),
            "dist_AP_std": dist_AP.std().item(),
            "dist_AP_min": dist_AP.min().item(),
            "dist_AP_max": dist_AP.max().item(),
            "dist_AN_mean": dist_AN.mean().item(),
            "dist_AN_std": dist_AN.std().item(),
            "dist_AN_min": dist_AN.min().item(),
            "dist_AN_max": dist_AN.max().item(),
        }

    def _batch_all(
        self,
        distance_matrix: torch.Tensor,
        mask_pos: torch.Tensor,
        mask_neg: torch.Tensor,
    ) -> dict:

        dist_AP = distance_matrix[mask_pos.bool()]
        dist_AN = distance_matrix[mask_neg.bool()]
        return self._build_loss_dict(dist_AP, dist_AN, mask_pos)

    def _batch_hard(
        self,
        distance_matrix: torch.Tensor,
        mask_pos: torch.Tensor,
        mask_neg: torch.Tensor,
    ) -> dict:

        dist_AP = self._mine_positives(distance_matrix, mask_pos)

        if self.negative_mining_mode == "hard":
            dist_AN, _ = triplet_mining.hard_negative_mining(distance_matrix, mask_neg)
        else:
            dist_AN, _ = triplet_mining.semi_hard_negative_mining(
                distance_matrix, dist_AP, mask_neg, self.margin
            )

        return self._build_loss_dict(dist_AP, dist_AN, mask_pos)

    def _batch_class_wise_hard(
        self,
        distance_matrix: torch.Tensor,
        mask_pos: torch.Tensor,
    ) -> dict:

        dist_AP = self._mine_positives(distance_matrix, mask_pos)

        # dist_AN: (B, P), valid_mask: (B, P)
        dist_AN, valid_mask = triplet_mining.class_wise_hard_negative_mining(
            distance_matrix, P=self.p, K=self.k
        )

        return self._build_loss_dict(dist_AP, dist_AN[valid_mask.bool()], mask_pos)
