import torch

from .info_nce import InfoNCELoss


class SupConLoss(InfoNCELoss):

    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False):

        super(SupConLoss, self).__init__(
            temperature=temperature, learn_temperature=learn_temperature
        )

    def forward(self, embeddings: torch.Tensor, clique_ids: torch.Tensor) -> dict:

        mask_pos, mask_neg = self._get_pos_neg_masks(clique_ids=clique_ids)
        loss_dict = self._get_loss_dict(embeddings, mask_pos, mask_neg)
        loss_dict["loss"] = loss_dict["loss"].mean()

        return loss_dict


class SupDCLLoss(SupConLoss):

    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False):

        super(SupDCLLoss, self).__init__(
            temperature=temperature, learn_temperature=learn_temperature
        )

    def _sum_denominator(
        self, logits: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor
    ) -> torch.Tensor:
        """In the SupConLoss, the denominator is the sum of the similarity matrix
        over the positive and negative samples. In the SupDCLLoss, the denominator
        is the sum of the similarity matrix over only the negative samples."""

        # Sum the denominator over only the negative samples
        denom = torch.sum(logits.exp() * mask_neg, dim=1, keepdim=True)

        return denom
