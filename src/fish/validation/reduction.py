import torch


@torch.no_grad()
def reduce_chunk_distances(
    D: torch.Tensor,
    sizes_batch: list[int],
    sizes: list[int],
) -> torch.Tensor:
    """
    D: (Qc, DBc) where Qc=sum(sizes_batch), DBc=sum(sizes)
    Returns: (B, N) B=num query tracks in the batch, N number of database tracks
    """

    device = D.device
    B = len(sizes_batch)
    N = len(sizes)

    assert D.size(0) == sum(sizes_batch), "Sizes and D do not match (rows)"
    assert D.size(1) == sum(sizes), "Sizes and D do not match (cols)"

    col_gid = torch.repeat_interleave(
        torch.arange(N, device=device), torch.tensor(sizes, device=device)
    )  # segments indexed by col_gid[i] and sizes[i] belong to the same database track
    row_gid = torch.repeat_interleave(
        torch.arange(B, device=device), torch.tensor(sizes_batch, device=device)
    )  #  segments indexed by row_gid[i] and sizes_batch[i] belong to the same query track

    def group_reduce_cols(x: torch.Tensor) -> torch.Tensor:
        # x: (M, Nc) -> out: (M, N) reducing columns by col_gid
        out = torch.full((x.size(0), N), float("inf"), dtype=x.dtype, device=device)
        out.scatter_reduce_(
            1, col_gid.expand(x.size(0), -1), x, reduce="amin", include_self=True
        )
        return out

    def group_reduce_rows(x: torch.Tensor) -> torch.Tensor:
        # x: (M, N) -> out: (B, N) reducing rows by row_gid
        out = torch.full((B, x.size(1)), float("inf"), dtype=x.dtype, device=device)
        out.scatter_reduce_(
            0,
            row_gid.view(-1, 1).expand(-1, x.size(1)),
            x,
            reduce="amin",
            include_self=True,
        )
        return out

    # NOTE: Only best reduction is possible currently
    D = group_reduce_cols(D)  # (M, N)
    D = group_reduce_rows(D)  # (B, N)

    return D
