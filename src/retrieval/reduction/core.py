import torch


def best_reduction(D, largest: bool):
    _, _, L_max = D.shape

    D_flat = D.flatten(1, 2)  # (N_batch, L_Q*L_max)
    if largest:
        values, best_col = D_flat.max(dim=1)
    else:
        values, best_col = D_flat.min(dim=1)

    # Map to local coordinates
    coords = torch.stack((best_col // L_max, best_col % L_max), 1)

    return values, coords


def best_r_reduction(
    D: torch.Tensor,  # (N_batch, L_Q, L_max)
    largest: bool,
    r: int = 10,
):
    _, _, L_max = D.shape
    D_flat = D.flatten(1, 2)  # (N_batch, L_Q*L_max)

    # Take best-r segments per track
    r = min(r, D_flat.size(1))
    values, indices = torch.topk(D_flat, k=r, dim=1, largest=largest)

    # Find the coordinate of the best matching segment pair per track
    best_col = indices[:, 0]  # Best single hit per row (N_batch,)
    coords = torch.stack((best_col // L_max, best_col % L_max), 1)

    # Aggregate the r values per track over finite entries only
    finite = torch.isfinite(values)  # (N_batch, k)
    cnt = finite.sum(dim=1)  # (# finite per row)
    values.masked_fill_(~finite, 0.0)
    values = values.sum(dim=1) / cnt

    return values, coords


def bpwor_r_reduction(
    D: torch.Tensor,  # (N_batch, L_Q, L_max)
    largest: bool,
    r: int = 10,
):

    fill = float("-inf") if largest else float("inf")

    N, L_Q, L_max = D.shape
    _idx = torch.arange(N, device=D.device)
    r = min(r, L_Q, L_max)

    values = torch.empty((N, r), device=D.device, dtype=D.dtype)
    coords = torch.empty((N, 2), device=D.device, dtype=torch.int64)
    for _r in range(r):
        D_flat = D.flatten(1, 2)  # (N_batch, L_Q*L_max)

        # Find best pair (q,s) per track
        if largest:
            values_r, indices_r = D_flat.max(dim=1)  # (N_batch,), (N_batch,)
        else:
            values_r, indices_r = D_flat.min(dim=1)
        values[:, _r] = values_r

        # Map to local coordinates
        coords_r = torch.stack((indices_r // L_max, indices_r % L_max), 1)

        # Store the best (q,s) pair's coordinates
        if _r == 0:
            coords = coords_r

        # Mask out all the elements in the row and column of (q,s)
        # NOTE: Due to slicingm this SHOULD be as fast as .index_put_
        D[_idx, coords_r[:, 0], :] = fill
        D[_idx, :, coords_r[:, 1]] = fill

    # Aggregate the r values per track over finite entries only
    finite = torch.isfinite(values)  # (N_batch, k)
    cnt = finite.sum(dim=1)  # (# finite per row)
    values.masked_fill_(~finite, 0.0)
    values = values.sum(dim=1) / cnt

    return values, coords
