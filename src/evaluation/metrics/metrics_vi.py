import numpy as np


def average_precision(relevance: np.ndarray, n_relevant: int) -> float:
    """Calculates the average precision (AP) for a single query. It can be used
    on the full list of results or a cut-off at K. In the case of a cut-off,
    it is assumed that the relevance array has been cut-off already. We decided to
    penalize AP for bad choice of K for consistency with the rest, i.e. no
    adjustment of n_relevant."""

    assert n_relevant > 0, "Number of relevant items must be greater than 0."

    ranks = np.arange(1, len(relevance) + 1)
    prec_at_k = np.cumsum(relevance) / ranks
    ap_at_n = (1 / n_relevant) * np.sum(prec_at_k * relevance)
    ap_at_n = float(ap_at_n)

    return ap_at_n


def normalized_average_rank(
    relevance: np.ndarray,
    n_relevant: int,
    database_size: int,
    biased: bool = False,
) -> float:

    assert n_relevant == np.sum(relevance)

    n = len(relevance)
    assert n_relevant <= n
    assert n == database_size - 1, "Database size does not match relevance length."
    # NOTE: in some use cases n == database_size is possible (without filtering the results)

    sum_of_perfect_ranks = n_relevant * (n_relevant + 1) // 2  # faster
    # sum_of_perfect_ranks = np.sum(np.arange(1, n_relevant + 1))
    sum_of_actual_ranks = np.sum(relevance * np.arange(1, n + 1))
    deviation = sum_of_actual_ranks - sum_of_perfect_ranks
    if biased:
        nar = 1 / (n_relevant * n) * deviation
    else:
        nar = (
            1 / (n_relevant * (n - n_relevant)) * deviation if n != n_relevant else 0.0
        )

    nar = float(100 * nar)

    return nar


def normalized_average_rank_at_K(
    relevance: np.ndarray,
    n_relevant: int,
    database_size: int,
    biased: bool = False,
) -> float:
    """Expects the relevance array to be cut-off at K already. If you want to use
    the full list, use normalized_average_rank() function above."""

    assert database_size >= n_relevant > 0
    N = database_size - 1  # exclude the query itself

    # Perfect ranking assumes all relevant items are at the top, regardless of K
    sum_of_perfect_ranks = n_relevant * (n_relevant + 1) // 2  # faster
    # sum_of_perfect_ranks = np.sum(np.arange(1, n_relevant + 1))

    sum_of_actual_ranks = np.sum(relevance * np.arange(1, int(relevance.size) + 1))

    # Simulate the worst case: put the remainder relevant items at the end
    n_relevant_top_k = int(np.sum(relevance))
    remainder_relevant = n_relevant - n_relevant_top_k
    if remainder_relevant > 0:
        penalty = remainder_relevant * (2 * N - remainder_relevant + 1) / 2
        # (2 * N - remainder_relevant + 1) / 2 is arithmetics
        sum_of_actual_ranks += penalty

    deviation = sum_of_actual_ranks - sum_of_perfect_ranks

    if biased:
        nar = 1 / (n_relevant * N) * deviation
    else:
        nar = (
            1 / (n_relevant * (N - n_relevant)) * deviation if n_relevant != N else 0.0
        )

    nar = float(100 * nar)

    return nar


def recall(relevance: np.ndarray, n_relevant: int) -> float:
    """Calculates the recall for a single query. It can be used
    on the full list of results or a cut-off at K. In the case of a cut-off,
    it is assumed that the relevance array has been cut-off already."""

    assert n_relevant > 0, "Number of relevant items must be greater than 0."

    R_K = float(np.sum(relevance) / n_relevant)

    return R_K


def version_id_metrics(
    q_fstem: str,
    result_list: list[dict],
    gt: dict,
    db_size: int,
    top_N: int | None = None,
) -> dict:

    assert len(result_list) > 0, "Result list must not be empty."

    q_gt = gt[q_fstem]
    q_clique_id = q_gt["clique_id"]
    n_relevant = len(q_gt["valid_other_version_yt_ids"])

    top_k = len(result_list)

    # Get the relevance for each candidate in the result list.
    # Find the position of the query's master track in the result list and
    # remove it from the list.
    relevance = np.zeros(len(result_list))
    _pos = None
    w = 0
    for j, candidate in enumerate(result_list):
        c_fstem = candidate["candidate_fstem"]
        c_gt = gt[c_fstem]
        c_clique_id = c_gt["clique_id"]
        if c_fstem == q_fstem:
            _pos = j
        else:
            if c_clique_id == q_clique_id:
                relevance[w] = 1
            w += 1
    if _pos is None:
        print(f"Warning: Query {q_fstem} not found in the result list.")
    relevance = relevance[:w]

    # Calculate metrics
    AP_K = average_precision(relevance, n_relevant)
    if top_N is None:
        NAR_K = normalized_average_rank(relevance, n_relevant, db_size - 1)
    else:
        NAR_K = normalized_average_rank_at_K(relevance, n_relevant, db_size)
    R_K = recall(relevance, n_relevant)

    metrics = {
        "AP@K": AP_K,
        "NAR@K": NAR_K,
        "Recall@K": R_K,
        "K": top_k,
        "reference_position": _pos,
    }

    return metrics


def aggregate_version_id_metrics(eval_metrics: list[dict]) -> dict:
    """Results must be sorted descending w.r.t similarity."""

    AP_K = np.array([m["AP@K"] for m in eval_metrics])
    NAR_K = np.array([m["NAR@K"] for m in eval_metrics])
    R_K = np.array([m["Recall@K"] for m in eval_metrics])
    top_k = np.array([m["K"] for m in eval_metrics])  # unique tracks retrieved
    raw_reference_position = [m["reference_position"] for m in eval_metrics]

    top_k_min = int(np.min(top_k))
    top_k_avg = int(np.mean(top_k))
    top_k_med = int(np.median(top_k))
    top_k_max = int(np.max(top_k))
    print("Top-k statistics (Unique number of tracks):")
    print(f"Min: {top_k_min}")
    print(f"Avg: {top_k_avg}")
    print(f"Med: {top_k_med}")
    print(f"Max: {top_k_max}")

    R_K_min = round(float(np.min(R_K)), 3)
    R_K_avg = round(float(np.mean(R_K)), 3)
    R_K_med = round(float(np.median(R_K)), 3)
    R_K_max = round(float(np.max(R_K)), 3)
    print("Recall@K statistics:")
    print(f"Min: {R_K_min:.3f}")
    print(f"Avg: {R_K_avg:.3f}")
    print(f"Med: {R_K_med:.3f}")
    print(f"Max: {R_K_max:.3f}")

    # Filter out None values for reference_position
    n_none = sum(x is None for x in raw_reference_position)
    reference_pos = np.array([x for x in raw_reference_position if x is not None])
    print("Reference track's position statistics:")
    print(f"{n_none} None values out of {len(raw_reference_position)}")
    ref_pos_min = int(np.min(reference_pos))
    ref_pos_avg = round(float(np.mean(reference_pos)), 2)
    ref_pos_med = round(float(np.median(reference_pos)), 2)
    ref_pos_max = int(np.max(reference_pos))
    print(f"Min: {ref_pos_min}")
    print(f"Avg: {ref_pos_avg:.2f}")
    print(f"Med: {ref_pos_med:.2f}")
    print(f"Max: {ref_pos_max}")

    print(f"{len(np.where(AP_K==0)[0]):,} queries returned AP@K=0")
    print(f"{len(np.where(AP_K==1)[0]):,} queries returned AP@K=1")

    print(f"{len(np.where(NAR_K==100)[0]):,} queries returned NAR@K=100")
    print(f"{len(np.where(NAR_K==0)[0]):,} queries returned NAR@K=0")

    M_AP_K, CI_AP_K = compute_sample_ci(AP_K)
    M_AP_K = np.round(M_AP_K, 3)
    CI_AP_K = np.round(CI_AP_K, 3)
    print(f"    AP@K: {M_AP_K} ± {CI_AP_K}")

    M_NAR_K, CI_NAR_K = compute_sample_ci(NAR_K)
    M_NAR_K = np.round(M_NAR_K, 2)
    CI_NAR_K = np.round(CI_NAR_K, 2)
    print(f"   NAR@K: {M_NAR_K} ± {CI_NAR_K}")

    M_R_K, CI_R_K = compute_sample_ci(R_K)
    M_R_K = np.round(M_R_K, 3)
    CI_R_K = np.round(CI_R_K, 3)
    print(f"Recall@K: {M_R_K} ± {CI_R_K}")

    metrics = {
        "M-AP@K": M_AP_K,
        "CI-AP@K": CI_AP_K,
        "M-NAR@K": M_NAR_K,
        "CI-NAR@K": CI_NAR_K,
        "M-Recall@K": M_R_K,
        "CI-Recall@K": CI_R_K,
        "K-min": top_k_min,
        "K-avg": top_k_avg,
        "K-med": top_k_med,
        "K-max": top_k_max,
        "R@K-min": R_K_min,
        "R@K-avg": R_K_avg,
        "R@K-med": R_K_med,
        "R@K-max": R_K_max,
        "ref_pos_none": n_none,
        "ref_pos_min": ref_pos_min,
        "ref_pos_avg": ref_pos_avg,
        "ref_pos_med": ref_pos_med,
        "ref_pos_max": ref_pos_max,
    }

    return metrics


def compute_sample_ci(measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    assert measurements.ndim == 1

    n = len(measurements)
    assert n > 0, "Measurements are empty."

    mean = measurements.mean()
    margin = 1.96 * measurements.std(ddof=1) / np.sqrt(n)

    return mean, margin
