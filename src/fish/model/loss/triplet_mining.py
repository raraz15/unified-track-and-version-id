from typing import Tuple

import torch

############################### Random Sampling ###############################


def random_positive_sampling(
    distance_matrix: torch.Tensor, mask_pos: torch.Tensor, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    if verbose:
        assert mask_pos.nelement() > 0, "mask_neg must be non-empty"
        assert torch.all(mask_pos.sum(1) >= 1), "Some anchors have no positives!"

    # Get the indices of the positive samples for each anchor point
    positive_indices = torch.multinomial(mask_pos, 1)

    # Get the distances between the anchors and their positive samples
    anchor_pos_distances = torch.gather(distance_matrix, 1, positive_indices)

    return anchor_pos_distances.squeeze(1), positive_indices.squeeze(1)


def random_negative_sampling(
    distance_matrix: torch.Tensor, mask_neg: torch.Tensor, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    if verbose:
        assert mask_neg.nelement() > 0, "mask_neg must be non-empty"
        assert torch.all(mask_neg.sum(1) >= 1), "Some anchors have no negatives!"

    # Get the indices of the negative samples for each anchor point
    negative_indices = torch.multinomial(mask_neg, 1)

    # Get the distances between the anchors and their negative samples
    anchor_neg_distances = torch.gather(distance_matrix, 1, negative_indices)

    return anchor_neg_distances.squeeze(1), negative_indices.squeeze(1)


############################### Hard Mining ###############################


def hard_positive_mining(
    distance_matrix: torch.Tensor, mask_pos: torch.Tensor, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    if verbose:
        assert mask_pos.nelement() > 0, "mask_pos must be non-empty"
        assert torch.all(mask_pos.sum(1) >= 1), "Some anchors have no positives!"

    # Select the hardest positive for each anchor
    anchor_pos_distances, positive_indices = torch.max(distance_matrix * mask_pos, 1)

    return anchor_pos_distances, positive_indices


def hard_negative_mining(
    distance_matrix: torch.Tensor, mask_neg: torch.Tensor, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    if verbose:
        assert mask_neg.nelement() > 0, "mask_neg must be non-empty"
        assert torch.all(mask_neg.sum(1) >= 1), "Some anchors have no negatives!"

    # Modify the distance matrix to only consider the negative samples
    masked_distances = distance_matrix.masked_fill(mask_neg == 0, float("inf"))

    # Get the indices of the hardest negative samples for each anchor point
    anchor_neg_distances, negative_indices = torch.min(masked_distances, 1)

    return anchor_neg_distances, negative_indices


############################### Semi-hard Mining ###############################


def semi_hard_negative_mining(
    distance_matrix: torch.Tensor,
    dist_AP: torch.Tensor,
    mask_neg: torch.Tensor,
    margin: float,
    mode_non_empty: str = "hard",
    mode_empty: str = "hard",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if verbose:
        assert mode_non_empty in {
            "hard",
            "random",
        }, "mode_non_empty must be either 'hard' or 'random'"
        assert mode_empty in {
            "hard",
            "random",
        }, "mode_empty must be either 'hard' or 'random'"
        assert margin > 0, "margin must be greater than 0"
        assert dist_AP.ndim == 1, "dist_AP must be a 1D tensor"

    # Initialize the tensors
    dist_AN = torch.zeros_like(dist_AP)
    negative_indices = torch.zeros_like(dist_AP, dtype=torch.long)

    # Get the region for semi-hard negatives
    mask_semi_hard_neg = (
        (dist_AP.unsqueeze(1) <= distance_matrix)
        & (distance_matrix < (dist_AP.unsqueeze(1) + margin))
        & mask_neg.bool()
    ).float()

    # Find which positive pairs have at least one semi-hard negative
    empty_hollow_sphere = mask_semi_hard_neg.sum(1) == 0
    non_empty_hollow_sphere = ~empty_hollow_sphere

    # If there are positive pairs with semi-hard negatives
    if non_empty_hollow_sphere.any():
        if mode_non_empty == "hard":
            # choose the hardest examples in the corresponding hollow-spheres
            anchor_neg_distances_with, negative_indices_with = hard_negative_mining(
                distance_matrix[non_empty_hollow_sphere],
                mask_semi_hard_neg[non_empty_hollow_sphere],
                verbose=verbose,
            )
        else:
            # choose random examples in the corresponding hollow-spheres
            anchor_neg_distances_with, negative_indices_with = random_negative_sampling(
                distance_matrix[non_empty_hollow_sphere],
                mask_semi_hard_neg[non_empty_hollow_sphere],
                verbose=verbose,
            )
        # Write the results
        dist_AN[non_empty_hollow_sphere] = anchor_neg_distances_with
        negative_indices[non_empty_hollow_sphere] = negative_indices_with

    # If there are positive pairs without semi-hard negatives
    if empty_hollow_sphere.any():
        # We use the entire batch for these cases
        if mode_empty == "hard":
            # Resort to hard negatives
            anchor_neg_distances_without, negative_indices_without = (
                hard_negative_mining(
                    distance_matrix[empty_hollow_sphere],
                    mask_neg[empty_hollow_sphere],
                    verbose=verbose,
                )
            )
        else:
            # Resort to random negatives
            anchor_neg_distances_without, negative_indices_without = (
                random_negative_sampling(
                    distance_matrix[empty_hollow_sphere],
                    mask_neg[empty_hollow_sphere],
                    verbose=verbose,
                )
            )
        # Write the results
        dist_AN[empty_hollow_sphere] = anchor_neg_distances_without
        negative_indices[empty_hollow_sphere] = negative_indices_without

    return dist_AN, negative_indices


########################## Class-wise Hard Mining #############################


def class_wise_hard_negative_mining(
    distance_matrix: torch.Tensor,
    P: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each anchor, mine the hardest negative from each negative class.
    Assumes the batch is PK-sampled: P classes, K samples each, ordered by class.

    Returns:
        dist_AN:    (B, P) — hardest negative distance per class;
                    0 for entries where the anchor belongs to that class.
        valid_mask: (B, P) — 1 where the triplet is valid
                    (anchor class != negative class).
    """
    B = P * K

    # (B, B) -> (B, P, K): for each anchor, group columns by class
    dist_by_class = distance_matrix.view(B, P, K)

    # Hardest (closest) negative per class: min over K samples
    dist_AN = dist_by_class.amin(dim=2)  # (B, P)

    # valid_mask: anchor i is NOT in class c
    # anchor i belongs to class i // K
    anchor_classes = torch.arange(P, device=distance_matrix.device).repeat_interleave(
        K
    )  # (B,)
    all_classes = torch.arange(P, device=distance_matrix.device)  # (P,)
    valid_mask = (
        anchor_classes.unsqueeze(1) != all_classes.unsqueeze(0)
    ).float()  # (B, P)

    dist_AN = dist_AN * valid_mask

    return dist_AN, valid_mask


################################# Easy Mining #################################


def easy_positive_mining(
    distance_matrix: torch.Tensor, mask_pos: torch.Tensor, verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    if verbose:
        assert mask_pos.nelement() > 0, "mask_pos must be non-empty"
        assert torch.all(mask_pos.sum(1) >= 1), "Some anchors have no positives!"

    # Modify the distance matrix to only consider the positive samples
    masked_distances = distance_matrix.masked_fill(mask_pos == 0, float("inf"))

    # Select the easiest positive for each anchor
    anchor_pos_distances, positive_indices = torch.min(masked_distances, 1)

    return anchor_pos_distances, positive_indices
