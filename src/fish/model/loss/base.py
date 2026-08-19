from typing import Tuple

import torch

from src.common.tensor_op import create_class_matrix


def create_pos_neg_masks(
    clique_ids_q: torch.Tensor, clique_ids_db: torch.Tensor, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    mask_pos = create_class_matrix(clique_ids_q, clique_ids_db).float()
    mask_neg = 1 - mask_pos

    mask_pos.fill_diagonal_(0)  # An anchor cannot be positive with itself

    if verbose:
        assert not torch.any(
            mask_pos + mask_neg > 1
        ), "Some anchors are both positive and negative!"
        assert torch.all(mask_pos.sum(1) >= 1), f"Some anchors have no positives!"
        assert torch.all(mask_neg.sum(1) >= 1), f"Some anchors have no negatives!"

    return mask_pos, mask_neg


class BaseLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def _get_pos_neg_masks(
        self, clique_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mask_pos, mask_neg = create_pos_neg_masks(
            clique_ids_q=clique_ids,
            clique_ids_db=clique_ids,
        )

        return mask_pos, mask_neg
