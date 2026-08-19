import torch

from src.common.tensor_op import pairwise_distance_matrix, create_class_matrix

from .base import BaseLoss


def create_ssl_masks(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """In SSL, class labels are not used."""

    N = embeddings.size(0)
    assert N % 2 == 0, "Batch size must be even"

    labels = torch.arange(N // 2, device=embeddings.device).repeat_interleave(2)
    mask_pos = create_class_matrix(labels, labels).float()
    mask_neg = 1 - mask_pos
    mask_pos.fill_diagonal_(0)  # A sample cannot be positive with itself

    return mask_pos, mask_neg


class InfoNCELoss(BaseLoss):
    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False):

        super().__init__()

        assert temperature > 0, "Temperature must be greater than 0"
        self.learn_temperature = learn_temperature
        self.temperature = torch.nn.Parameter(
            torch.tensor(temperature), requires_grad=learn_temperature
        )

    def forward(self, embeddings: torch.Tensor, clique_ids: torch.Tensor) -> dict:

        # InfoNCE uses SSL masks
        mask_pos, mask_neg = create_ssl_masks(embeddings)
        assert torch.all(mask_pos.sum(1) == 1)
        loss_dict = self._get_loss_dict(embeddings, mask_pos, mask_neg)
        loss_dict["loss"] = loss_dict["loss"].mean()

        return loss_dict

    def _get_loss_dict(
        self, embeddings: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor
    ) -> dict:

        # Get the logits and raw similarities
        sim_matrix, logits = self._compute_logits(embeddings)

        # Compute the denominator
        denom = self._sum_denominator(logits, mask_pos, mask_neg)

        # Compute the log_probabilities for each positive pair
        log_prob = (logits - torch.log(denom)) * mask_pos

        # Count the number of positives for each anchor
        # NOTE: in SSL, there is only one positive pair for each anchor,
        # but we keep this code for consistency with the SupConLoss
        n_pos = mask_pos.sum(1)

        # Calculate the loss for each anchor
        loss = (-1 / n_pos) * torch.sum(log_prob, dim=1)

        # Extract similarity and squared euclidean distance stats
        # For L2-normalized embeddings: ||a - b||^2 = 2 - 2 * cos(a, b)
        dist_AP = 2 - 2 * sim_matrix[mask_pos.bool()]
        dist_AN = 2 - 2 * sim_matrix[mask_neg.bool()]

        loss_dict = {
            "loss": loss,
            "max_class_size": n_pos.int().max().item() + 1,
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

    def _compute_logits(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Compute the pairwise similarity matrix between the embeddings
        # NOTE: we assume that the embeddings are already l2-normalized
        similarity_matrix = pairwise_distance_matrix(embeddings, mode="dotsim")

        if self.learn_temperature:
            t = self.temperature.exp()
        else:
            t = 1 / self.temperature
        scaled = t * similarity_matrix

        # NOTE: for numerical stability its important to subtract the maximum
        # logit from each row before exponentiating
        logits_max = torch.amax(scaled, dim=1, keepdim=True).detach()
        logits = scaled - logits_max

        return similarity_matrix, logits

    def _sum_denominator(
        self, logits: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor
    ) -> torch.Tensor:

        # Sum the denominator over all real pairs
        denom = torch.sum(logits.exp() * (mask_pos + mask_neg), dim=1, keepdim=True)

        return denom


class DCL(InfoNCELoss):

    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False):

        super(DCL, self).__init__(
            temperature=temperature, learn_temperature=learn_temperature
        )

    def _sum_denominator(
        self, logits: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor
    ) -> torch.Tensor:
        """In the DCL, the denominator is the sum of the similarity matrix over
        only the negative samples."""

        # Sum the denominator over only the negative samples
        denom = torch.sum(logits.exp() * mask_neg, dim=1, keepdim=True)

        return denom
