import torch


def build_dense_partial_D(
    I: torch.Tensor,
    S: torch.Tensor,
    track_lengths: torch.Tensor,
    track_starts: torch.Tensor,
    largest: bool,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense distance matrix for the tracks that appear in the results.

    I: (L_Q, k) int64 global segment indices
    S: (L_Q, k) float32 scores
    track_lengths: (N,) int32 track lengths
    track_starts: (N+1,) int32 track boundaries in the global segments
    largest: if True, larger scores are better (similarities). If False, smaller
    scores are better (distances). E.g. for metric="L2", largest=False.
    verbose: if True, print some info about invalid indices

    Returns: D of shape (N, L_Q, L_max) filled with inf (or -inf for similarities),
            and filled at (t,q,s) where we have hits.
            Also returns the track IDs (global indices) that correspond to the rows of D.
    """

    device = I.device
    L_Q, _ = I.shape

    # Mask out padded/invalid indices that can come from IVF
    valid = S.isfinite() & (I >= 0) & (I < track_starts[-1])
    if not valid.all():
        if verbose:
            n_invalid = (~valid).sum().item()
            print(f"Found {n_invalid:,} invalid items ({100*n_invalid/I.numel():.3f}).")
        q_idx_full = torch.arange(L_Q, device=device).unsqueeze(1).expand_as(I)
        I = I[valid]
        S = S[valid]
        q_idx = q_idx_full[valid]
    else:
        q_idx = torch.arange(L_Q, device=device).unsqueeze(1).expand_as(I).reshape(-1)
        I = I.reshape(-1)
        S = S.reshape(-1)

    # Map global segment index to track ID
    # TODO: more efficient way?
    # track_idx_aux = torch.arange(track_lengths.numel(), device=device).repeat_interleave(track_lengths)
    # t_idx = track_idx_aux[I]
    t_idx = torch.bucketize(I, track_starts, right=True) - 1  # (num_hits,)

    # Map global segment index to segment-in-track
    s_idx = I - track_starts[t_idx]  # (num_hits,)

    # Find the unique number of tracks and the inverse indices.
    # We use the inverse indices to fill D
    t_idx_unique, t_idx_inverse = torch.unique(t_idx, sorted=False, return_inverse=True)

    # Number of unique tracks in the results
    N_batch = t_idx_unique.numel()
    # Length of the longest track in the results
    L_batch_max = int(torch.max(track_lengths[t_idx_unique]).item())

    # Allocate D and fill
    fill = float("-inf") if largest else float("inf")
    D = torch.full((N_batch, L_Q, L_batch_max), fill, device=device, dtype=S.dtype)

    # Direct indexed assignment (no reduction). If duplicates exist, "last write wins".
    D.index_put_((t_idx_inverse, q_idx, s_idx), S, accumulate=False)

    return D, t_idx_unique
