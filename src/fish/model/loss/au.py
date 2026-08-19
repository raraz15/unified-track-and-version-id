import torch

from .base import BaseLoss


class AULoss(BaseLoss):
    def __init__(self, alpha: float = 2, gamma: float = 2, w_uniformity: float = 1):
        """Initialize the AULoss.

        Parameters:
        -----------
        alpha: float
            The exponent used in the alignment loss.
        gamma: float
            The scaling parameter used in the uniformity loss.
        w_alignment: float
            The weight of the alignment loss.
        w_uniformity: float
            The weight of the uniformity loss.
        """

        super().__init__()
        assert w_uniformity > 0, "Expected a positive weight for uniformity loss."

        self.alignment_loss = AlignmentLoss(alpha)
        self.uniformity_loss = UniformityLoss(gamma)
        self.w_uniformity = w_uniformity

    def forward(self, embeddings: torch.Tensor, clique_ids: torch.Tensor) -> dict:
        """

        Parameters:
        -----------
        embeddings: torch.Tensor
            2D tensor of shape (2*n, d) where n is the number of pairs and d is
            the dimensionality.

            NOTE: assumes that the tensor is already paired, i.e. torch.split(x, 2)
            groups into positive pairs.

        clique_ids: torch.Tensor
            Not used for this SSL loss, but kept for compatibility with other losses.

        version_ids: torch.Tensor
            Not used for this SSL loss, but kept for compatibility with other losses.

        Returns:
        --------
        loss_dict: {'loss': torch.Tensor, 'loss_alignment': torch.Tensor,
                    'loss_uniformity': torch.Tensor}
        """

        assert (
            embeddings.ndim == 2
        ), f"Expected 2D tensor, got {embeddings.ndim}D tensor."
        assert (
            embeddings.shape[0] % 2 == 0
        ), "Alignment&Uniformity loss supports two samples per class."

        loss_align = self.alignment_loss(embeddings)
        loss_uniformity = self.uniformity_loss(embeddings)

        # combine the losses by weighing them
        loss = loss_align + self.w_uniformity * loss_uniformity

        loss_dict = {
            "loss": loss,
            "loss_alignment": loss_align,
            "loss_uniformity": loss_uniformity,
        }

        return loss_dict


class AlignmentLoss(torch.nn.Module):

    def __init__(self, alpha: float = 2):
        """Initialize the AlignmentLoss.

        Parameters:
        -----------
        alpha: float
            The exponent used to scale the norm of the difference in alignment loss.
        """

        super(AlignmentLoss, self).__init__()
        assert alpha > 0, "Expected a positive exponent for the alignment loss."

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the alignment loss between two sets of latents. Expects the input to be
        grouped into positive pairs along the first dimension. C0,C0,C1,C1,...

        Parameters:
        -----------
        x: torch.Tensor
            2D tensor of shape (N, D) where N is the number of samples and D is the
            dimensionality.

        Returns:
        --------
        loss_dict: {'loss': torch.Tensor}
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."

        loss = (x[0::2] - x[1::2]).norm(p=2, dim=1).pow(self.alpha).mean()

        return loss


class UniformityLoss(torch.nn.Module):

    def __init__(self, gamma: float = 2):
        """Initialize the UniformityLoss.

        Parameters:
        -----------
        t: float
            The temperature used in the uniformity loss.
        """

        super().__init__()
        assert gamma > 0, "Expected a positive gamma for the uniformity loss."

        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the uniformity loss for the given set of latents. Average the uniformity loss
        over the two groups of samples (even and odd indices) to fit to memory.

        Parameters:
        -----------
        x: torch.Tensor
            2D tensor of shape (n, d) where n is the number of all samples and d is the
            dimensionality.
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."

        loss0 = torch.pdist(x[0::2], p=2).pow(2).mul(-self.gamma).exp().mean().log()
        loss1 = torch.pdist(x[1::2], p=2).pow(2).mul(-self.gamma).exp().mean().log()
        loss = (loss0 + loss1) / 2

        return loss
