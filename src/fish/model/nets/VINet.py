import torch
import torch.nn as nn

from src.common.tensor_op import l2_normalize
from src.common.audio import SAMPLE_RATE


class VINet(nn.Module):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        l2_normalize_encoding: bool = False,
        l2_normalize_projection: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.l2_normalize_encoding = l2_normalize_encoding
        self.l2_normalize_projection = l2_normalize_projection
        assert eps > 0, "Epsilon must be positive."
        self.eps = eps

        # To be defined in the child class
        self.front_end = None
        self.back_end = None
        self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project the input tensor to the space where the contrastive loss will
        be applied. Corresponds to z = Proj(Enc(x))."""

        x = self.encode(x)
        x = self.project(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor `x`. Corresponds to h = Enc(x)."""

        x = self.front_end(x)
        x = self.back_end(x)
        if self.l2_normalize_encoding:
            x = self._l2_normalize(x)
        return x

    def project(self, r: torch.Tensor) -> torch.Tensor:
        """Projects the representation tensor r. Corresponds to z = Proj(h)."""

        r = self.projection(r)
        if self.l2_normalize_projection:
            r = self._l2_normalize(r)
        return r

    def embed(
        self, x: torch.Tensor, layer: str, normalize: bool = False
    ) -> torch.Tensor:
        """Embeds the input tensor `x` at the specified layer. This tensor is used
        as the representation of the audio. Corresponds to h = Enc(x) or z = Proj(Enc(x)).
        You can choose to normalize the output tensor in addition to self.encode()'s and
        self.project()'s normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be embedded.
        layer : str
            Layer to embed the tensor at. Can be "enc" or "proj".
        normalize : bool
            Whether to L2 normalize the output tensor.

        Returns
        -------
        torch.Tensor
            Embedded tensor.
        """

        assert layer in {"enc", "proj"}, "Invalid layer. Must be 'enc' or 'proj'."

        x = self.encode(x)
        if layer == "proj":
            x = self.project(x)
        if normalize:
            x = self._l2_normalize(x)
        return x

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(x, eps=self.eps)
