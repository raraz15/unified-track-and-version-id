from typing import Optional

import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    assert x.dim() == 2, "Input tensor must be 2D"

    norms = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
    x = x / norms.clamp(min=eps)

    return x


def pairwise_distance_matrix(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    mode: str = "euc",
    p: Optional[int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:

    assert x.dim() == 2, "Input tensors must be 2D"
    assert y is None or y.dim() == 2, "Input tensors must be 2D"
    assert mode in (
        "fro",
        "nfro",
        "euc",
        "neuc",
        "sqeuc",
        "nsqeuc",
        "cos",
        "cossim",
        "dot",
        "dotsim",
    ), f"Unsupported mode: {mode}"

    if y is None:
        y = x

    if mode in ("fro", "nfro"):
        assert p is not None, f"p must be specified for {mode}"
        dist = torch.cdist(x, y, p=p).clamp(0)
        if mode == "nfro":
            dist = dist / (x.size(-1) ** (1 / p))
    elif mode in ("euc", "neuc"):
        dist = pairwise_euclidean_distance(x, y, squared=False)
        if mode == "neuc":
            dist = dist / (x.size(-1) ** 0.5)
    elif mode in ("sqeuc", "nsqeuc"):
        dist = pairwise_euclidean_distance(x, y, squared=True)
        if mode == "nsqeuc":
            dist = dist / x.size(-1)
    elif mode in ("cos", "cossim", "dot", "dotsim"):
        if mode in ("cos", "cossim"):
            x = l2_normalize(x, eps=eps)
            y = l2_normalize(y, eps=eps)
        dist = torch.matmul(x, y.T)
        if mode in ("cos", "dot"):
            dist = 1 - dist
    else:
        raise NotImplementedError

    return dist


def pairwise_euclidean_distance(
    x: torch.Tensor, y: torch.Tensor, squared: bool = False, eps: float = 1e-6
) -> torch.Tensor:

    # ||x - y||^2 = ||x||^2 - 2<x, y> + ||y||^2
    squared_x = x.pow(2).sum(1).view(-1, 1)
    squared_y = y.pow(2).sum(1).view(1, -1)
    distance_matrix = squared_x - 2 * torch.mm(x, y.t()) + squared_y

    # Handle numerical instabilities by ensuring all distances are non-negative
    distance_matrix = torch.clamp(distance_matrix, min=0)

    # Derivative of the square root function at 0 is infinite
    if not squared:
        mask = (distance_matrix == 0.0).type_as(distance_matrix).detach()
        # Zeros become eps, non-zeros are untouched
        distance_matrix = distance_matrix + mask * eps
        # Now safe: sqrt(eps) is small but finite, gradient is large but finite
        distance_matrix = torch.sqrt(distance_matrix)
        # sqrt(eps)' go back to 0.0, non-zeros multiplied by 1.0
        distance_matrix = distance_matrix * (1.0 - mask)
        # The gradient for zero entries are now zero and not infinite

    return distance_matrix


def create_class_matrix(
    labels_q: torch.Tensor,
    labels_db: torch.Tensor,
) -> torch.Tensor:
    """Takes 1D tensors of integer class labels and creates a binary matrix indicating
    class membership.

    Parameters:
    -----------
    labels_q: torch.Tensor
        1D tensor of shape (B,) where labels[i] is the integer label of the i-th query item.
    labels_db: torch.Tensor
        1D tensor of shape (N,) where labels[i] is the integer label of the i-th database item.

    Returns:
    --------
    class_matrix: torch.Tensor
        2D Boolean tensor of shape (B, N) where class_matrix[i, j] is True
        if labels_q[i] == labels_db[j] and False otherwise.
    """

    assert labels_q.dim() == 1, f"labels_q must be a 1D tensor {labels_q.dim()}D"
    assert labels_db.dim() == 1, f"labels_db must be a 1D tensor {labels_db.dim()}D"

    class_matrix = labels_q.unsqueeze(1) == labels_db.unsqueeze(0)  # (B, N)
    return class_matrix
