import torch


@torch.inference_mode()
def reduce_to_track_level(
    S,
    I,
    track_lengths: torch.Tensor,  # (N,)
    track_starts: torch.Tensor,  # (N+1,) track boundaries in the global segments
    metric: str = "L2",
    verbose: bool = False,
) -> list[tuple[float, tuple[int, int], int]]:
    """Reduce segment-level matches to one summary segment per track. Returns a list
    of (score, (q_index, s_in_track), track_ids) tuples, sorted by descending
    similarity."""

    metric = metric.upper()
    assert metric in {"L2", "IP", "CS"}, f"Unknown metric: {metric}"
    largest = metric != "L2"

    total_segments = int(track_starts[-1].item())
    n_tracks = track_lengths.shape[0]

    # Convert S and I to torch tensors from cuVS output
    S = torch.as_tensor(S)
    I = torch.as_tensor(I)

    device = I.device

    # Gather only the valid scores and indices
    # TODO: Maybe 3 checks is overkill, S and I are large matrices (Q_L, k)
    valid = S.isfinite() & (I >= 0) & (I < total_segments)
    q_idx = valid.nonzero(as_tuple=True)[0]
    S_valid = S[valid].reshape(-1)
    I_valid = I[valid].reshape(-1)
    coords = torch.stack((q_idx, I_valid), dim=1)

    if verbose and not valid.all():
        n_invalid = I.numel() - S_valid.numel()
        print(f"Found {n_invalid:,} invalid items ({100*n_invalid/I.numel():.3f}).")

    if S_valid.numel() == 0:
        return []

    n_tracks = track_lengths.shape[0]
    n_matches = S_valid.numel()

    # Map global segment index to track ID
    track_ids = torch.bucketize(I_valid, track_starts, right=True) - 1  # (num_hits,)
    # TODO: more efficient / faster way? The one below takes too much memory, but its fast
    # track_idx_aux = torch.arange(track_lengths.numel(), dtype=torch.int32, device=device).repeat_interleave(track_lengths)
    # t_idx = track_idx_aux[I]

    # Sort the candidates by descending similarity (L2 ascending, IP decreasing)
    S_valid, sorted_indices = torch.sort(S_valid, descending=largest)
    track_ids = track_ids[sorted_indices]
    coords = coords[sorted_indices]

    # Find locations of the best segments of each track (the one with the smallest index)
    big = torch.iinfo(torch.int64).max
    pos = torch.arange(n_matches, device=device)
    first_pos = torch.full((n_tracks,), big, device=device)
    first_pos.scatter_reduce_(0, track_ids, pos, reduce="amin", include_self=True)
    # Filter out tracks that had no matches
    keep_pos = first_pos[first_pos < big]

    # Keep only the best segment per track
    S_valid = S_valid[keep_pos]
    track_ids = track_ids[keep_pos]
    coords = coords[keep_pos]

    # Sort tracks by best score (better first)
    S_valid, sorted_indices = torch.sort(S_valid, descending=largest)
    track_ids = track_ids[sorted_indices]
    coords = coords[sorted_indices]

    # Return as Python list of tuples
    return list(
        zip(
            S_valid.cpu().tolist(),
            coords.cpu().tolist(),
            track_ids.cpu().tolist(),
        )
    )
