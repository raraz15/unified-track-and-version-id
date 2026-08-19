import math
from typing import Optional

import torch

_metrics = ["AP", "R1", "Top1", "Top10", "Top100", "Top1000", "Top10000", "RR", "JNAR"]


@torch.inference_mode()
def normalized_average_rank(
    relevance: torch.Tensor,
    n_relevant: torch.Tensor,
    ranking: torch.Tensor,
    biased=False,
) -> torch.Tensor:
    k = ranking.shape[1]
    if biased:
        nar = (1 / (n_relevant * k)) * (
            torch.sum(relevance * ranking, 1) - (n_relevant * (n_relevant + 1) / 2)
        )
    else:
        nar = (100 / (n_relevant * (k - n_relevant))) * (
            torch.sum(relevance * ranking, 1) - (n_relevant * (n_relevant + 1) / 2)
        )
    return nar


@torch.inference_mode()
def normalized_average_rank_at_k(
    relevance: torch.Tensor,  # (B, k)
    n_relevant: torch.Tensor,  # (B, )
    ranking: torch.Tensor,  # (1, k)
    database_size: int,
    biased=False,
) -> torch.Tensor:

    assert (database_size >= n_relevant).all()
    assert (n_relevant > 0).all()
    assert database_size > ranking.shape[1]
    N = database_size - 1  # exclude the query itself

    # Perfect ranking assumes all relevant items are at the top, regardless of K
    # sum_of_perfect_ranks = torch.sum(torch.arange(1, n_relevant + 1))
    sum_of_perfect_ranks = n_relevant * (n_relevant + 1) / 2  # (B,)

    sum_of_actual_ranks = torch.sum(relevance * ranking, 1)  # (B, )

    # Simulate the worst case: put the remainder relevant items at the end
    n_relevant_top_k = torch.sum(relevance, 1)  # (B, )
    remainder_relevant = n_relevant - n_relevant_top_k  # (B, )
    penalty = remainder_relevant * (2 * N - remainder_relevant + 1) / 2  # (B, )
    sum_of_actual_ranks.add_(penalty)

    deviation = sum_of_actual_ranks - sum_of_perfect_ranks

    if biased:
        nar = 1 / (n_relevant * N) * deviation
    else:
        nar = 100 / (n_relevant * (N - n_relevant)) * deviation
        # NOTE if N==n_relevant this will explode

    return nar


@torch.inference_mode()
def average_precision(
    relevance: torch.Tensor, n_relevant: torch.Tensor, ranking: torch.Tensor
) -> torch.Tensor:
    prec_at_k = torch.cumsum(relevance, 1).div_(ranking)  # (B, k)
    ap = torch.sum(prec_at_k * relevance, 1).div_(n_relevant)  # (B,)
    return ap


@torch.inference_mode()
def rank_of_first_hit(
    relevance: torch.Tensor,  # (B, k)
    database_size: Optional[int] = None,
) -> torch.Tensor:

    hits = relevance > 0
    # For rows with any hit, argmax gives the index of the first True
    first_idx = torch.argmax(hits.to(torch.int32), dim=1)  # (B,)
    any_hits = hits.any(dim=1)  # (B,)

    # Rank of the first hit
    rank_first = first_idx + 1  # (B,)

    # Worst-case penalty rank
    if database_size is not None:
        worst = float(database_size - 1)
    else:
        worst = float(relevance.shape[1] + 1)  # fall back to k +1

    worst_rank = torch.full_like(rank_first, worst)

    # Use rank_first where there is a hit; otherwise use worst_rank
    tr = torch.where(any_hits, rank_first, worst_rank)
    return tr


def calculate_metrics(D: torch.Tensor, C: torch.Tensor, k: int) -> dict:
    """
    Calculate retrieval performance metrics for a given distance matrix and class matrix
    at the track-level.

    Parameters:
    -----------
    D: torch.tensor
        The distance matrix of shape (B, N) where B is the number of queries
        and N is the number of tracks.
    C: torch.tensor
        The class matrix of shape (B, N) where B is the number of queries and N is
        the number of tracks.
    k: int
        The number of similar tracks to consider for each query.

    Returns:
    --------
    metrics: dict Each key is associated with a tensor of shape (B,) where B is the
    number of queries in the batch.
    """

    assert D.dim() == 2, "Distance matrix must be 2D"
    db_size = D.shape[1]
    assert C.dim() == 2, "Class matrix must be 2D"
    assert D.size(0) == C.size(
        0
    ), "Distance and class matrices must have the same batch size"
    assert db_size == C.size(
        1
    ), "Distance and class matrices must have the same number of tracks"
    assert k > 0, "Number of similar tracks to consider must be positive"
    assert k <= db_size, f"You ask for more items then there is {k=}, {db_size}"

    # Number of relevant items (excluding itself) in the database for each query
    n_relevant = torch.sum(C, 1)  # (B,)
    assert torch.all(
        n_relevant > 0
    ), "There must be at least one relevant item for each query"

    # Ranking tensor 1,2,3,,,,,k
    ranking = torch.arange(1, k + 1, dtype=torch.float32, device=D.device).unsqueeze(
        0
    )  # (1, k)

    # For each embedding, find the indices of the k most similar embeddings
    _, spred = torch.topk(D, k, dim=1, largest=False)  # (B, k)
    # Get the relevance values of the k most similar embeddings
    relevance = torch.gather(C, 1, spred)  # (B, k)

    ap = average_precision(relevance, n_relevant, ranking)  # (B,)

    top1 = relevance[:, 0]  # (B,)
    top10 = relevance[:, :10].sum(1) if k >= 10 else None  # (B,)
    top100 = relevance[:, :100].sum(1) if k >= 100 else None  # (B,)
    top1000 = relevance[:, :1000].sum(1) if k >= 1000 else None  # (B,)
    top10000 = relevance[:, :10000].sum(1) if k >= 10000 else None  # (B,)

    if k == db_size - 1:
        tr = rank_of_first_hit(relevance)  # (B,)
        # nar = normalized_average_rank(
        #     relevance, n_relevant, ranking, biased=True
        # )  # (B,)
        jnar = normalized_average_rank(
            relevance, n_relevant, ranking, biased=False
        )  # (B,)

    else:
        tr = rank_of_first_hit(relevance, db_size)  # (B,)
        # nar = normalized_average_rank_at_k(
        #     relevance, n_relevant, ranking, db_size, biased=True
        # )  # (B,)
        jnar = normalized_average_rank_at_k(
            relevance, n_relevant, ranking, db_size, biased=False
        )  # (B,)

    return {
        "Top1": top1,
        "Top10": top10,
        "Top100": top100,
        "Top1000": top1000,
        "Top10000": top10000,
        "R1": tr,
        "AP": ap,
        "JNAR": jnar,
        "RR": 1 / tr,
    }


def aggregate_metrics(metrics: dict, confidence_interval: bool = True) -> dict:
    """This function aggregates the metrics and prints them to the console."""

    for key in list(metrics.keys()):
        if metrics[key] is None:
            del metrics[key]
            continue
        if key in ["Top1", "Top10", "Top100", "Top1000", "Top10000"]:
            metrics[key] = metrics[key].int().sum().item()
            print(f"{key:>8}: {metrics[key]:>8,}")
        elif key == "R1":
            MR1 = math.ceil(metrics.pop(key).mean().item())
            metrics[f"M-{key}"] = MR1
            print(f"{key:>8}: {MR1}")
        else:
            arr = metrics.pop(key)
            mean_val = round(arr.mean().item(), 3)
            metrics[f"M-{key}"] = mean_val
            if confidence_interval:
                # Calculate the confidence interval at a 95% confidence level
                std_val = arr.std().item()
                CI = round(1.96 * std_val / (len(arr) ** 0.5), 3)
                metrics[f"CI-{key}"] = CI
                print(f"{key:>8}: {mean_val:.3f} ± {CI:.3f}")
            else:
                print(f"{key:>8}: {mean_val:.3f}")

    return metrics


def composite_metrics(metrics: dict, monitor: list[str]) -> float:
    """This function calculates the composite metric based on the monitoring metrics."""

    assert len(monitor) > 1, "At least two monitoring metrics are required"

    comp_metric = []
    for m in monitor:
        if m == "M-NAR":
            comp_metric.append(1 - metrics[m])
        elif m == "M-JNAR":
            comp_metric.append(100 - metrics[m])
        elif m in {"M-AP", "M-RR"}:
            comp_metric.append(metrics[m])
        else:
            raise ValueError(f"{m} is not supported.")
    comp_metric = geometric_mean(comp_metric)

    print(f"Geometric Mean of {monitor}: {comp_metric:.4f}")
    return comp_metric


def geometric_mean(values):
    assert all(v > 0 for v in values), "All values must be positive"
    return math.pow(math.prod(values), 1 / len(values))
