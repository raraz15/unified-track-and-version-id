import torch
import torch.nn.functional as F

from src.common.tensor_op import pairwise_distance_matrix
import src.fish.model.loss.triplet_mining as triplet_mining

from .base import BaseLoss


class TripletLoss(BaseLoss):
    def __init__(
        self,
        margin: float = 0.3,
        positive_mining_mode: str = "hard",
        negative_mining_mode: str = "hard",
        squared_distance: bool = False,
        normalize_distance: bool = False,
        soft: bool = False,
    ):

        super().__init__()

        assert margin >= 0, "Margin must be greater or equal to 0."
        assert positive_mining_mode.lower() in [
            "random",
            "easy",
            "hard",
        ], "Positive mining mode must be either 'random', 'easy', or 'hard'."
        assert negative_mining_mode.lower() in [
            "random",
            "semi-hard",
            "hard",
        ], "Negative mining mode must be either 'random', 'semi-hard', or 'hard'."

        self.margin = margin
        self.positive_mining_mode = positive_mining_mode.lower()
        self.negative_mining_mode = negative_mining_mode.lower()
        self.soft = soft

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
        loss_dict = self._get_loss_dict(embeddings, mask_pos, mask_neg)
        loss_dict["loss"] = loss_dict["loss"].mean()

        return loss_dict

    def _get_loss_dict(
        self, embeddings: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor
    ) -> dict:

        # Compute the pairwise distance matrix between the samples
        distance_matrix = pairwise_distance_matrix(embeddings, mode=self.dist)

        # Mine the positives first then the negatives
        dist_AP = self._mine_positives(distance_matrix, mask_pos)
        dist_AN = self._mine_negatives(distance_matrix, dist_AP, mask_neg)

        # Compute the loss
        if self.soft:
            loss = F.softplus(dist_AP - dist_AN + self.margin)
        else:
            loss = F.relu(dist_AP - dist_AN + self.margin)

        loss_dict = {
            "loss": loss,
            "num_unsatisfied_anchors": int((loss > 0).sum().item()),
            "max_class_size": mask_pos.sum(1).int().max().item() + 1,
            "dist_AP_mean": dist_AP.mean().item(),
            "dist_AP_std": dist_AP.std().item(),
            "dist_AP_min": dist_AP.min().item(),
            "dist_AP_max": dist_AP.max().item(),
            "dist_AN_mean": dist_AN.mean().item(),
            "dist_AN_std": dist_AN.std().item(),
            "dist_AN_min": dist_AN.min().item(),
            "dist_AN_max": dist_AN.max().item(),
        }

        return loss_dict

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

    def _mine_negatives(
        self,
        distance_matrix: torch.Tensor,
        dist_AP: torch.Tensor,
        mask_neg: torch.Tensor,
    ) -> torch.Tensor:

        if self.negative_mining_mode == "random":
            dist_AN, _ = triplet_mining.random_negative_sampling(
                distance_matrix, mask_neg
            )
        elif self.negative_mining_mode == "hard":
            dist_AN, _ = triplet_mining.hard_negative_mining(distance_matrix, mask_neg)
        else:
            dist_AN, _ = triplet_mining.semi_hard_negative_mining(
                distance_matrix, dist_AP, mask_neg, self.margin
            )
        return dist_AN
