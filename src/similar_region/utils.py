import math

import torch


def r3(x):
    return round(float(x), 3)


def get_frames(
    x, length, step, dim=-1, pad_end=True, pad_mode="zeros", cut_mode="start"
):
    if pad_end:
        newlength = (
            max(int(math.ceil((x.size(dim) - length) / step)), 0) * step + length
        )
        x = force_length(
            x,
            newlength,
            dim=dim,
            pad_mode=pad_mode,
            cut_mode=cut_mode,
            allow_longer=False,
        )
    return x.unfold(dim, length, step)


def force_length(
    x, length, dim=-1, pad_mode="repeat", cut_mode="start", allow_longer=False
):
    assert pad_mode in ("repeat", "zeros", "crazy")
    assert cut_mode in ("start", "end", "random")
    # fast bypass
    if x.size(dim) == length or (x.size(dim) > length and allow_longer):
        return x
    # do otherwise
    aux = x.clone()
    while aux.size(dim) < length:
        if pad_mode == "repeat":
            aux = torch.cat([aux, x], dim=dim)
        elif pad_mode == "zeros":
            aux = torch.cat([aux, torch.zeros_like(x)], dim=dim)
        elif pad_mode == "crazy":
            r = torch.randint(0, 4, (1,)).item()
            if r == 0:
                aux = torch.cat([aux, x], dim=dim)
            elif r == 1:
                aux = torch.cat([x, aux], dim=dim)
            elif r == 2:
                aux = torch.cat([aux, torch.zeros_like(x)], dim=dim)
            elif r == 3:
                aux = torch.cat([torch.zeros_like(x), aux], dim=dim)
    if not allow_longer and aux.size(-1) > length:
        if dim != -1:
            aux = aux.transpose(dim, -1)
        if cut_mode == "start":
            aux = aux[..., :length]
        elif cut_mode == "end":
            aux = aux[..., -length:]
        elif cut_mode == "random":
            r = torch.randint(0, aux.size(-1) - length + 1, (1,)).item()
            aux = aux[..., r : r + length]
        if dim != -1:
            aux = aux.transpose(-1, dim)
    return aux
