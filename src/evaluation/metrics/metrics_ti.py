import numpy as np


def top_n_track_hit(ref_fstem: str, result_list: list, n: int) -> int:
    """Check if any of the top 10 candidates' tracks matches the ground truth reference track."""

    if len(result_list) < n:
        print(f"[WARNING]: result_list has fewer than {n} candidates.")

    for candidate in result_list[:n]:
        if candidate["candidate_fstem"] == ref_fstem:
            return 1
    return 0


def track_id_metrics(q_fstem: str, result_list: list, gt: dict) -> dict:

    ref_fstem = gt[q_fstem]["reference_fstem"]

    top1_track_hit = top_n_track_hit(ref_fstem, result_list, 1)
    if top1_track_hit:
        top10_track_hit = 1
    else:
        top10_track_hit = top_n_track_hit(ref_fstem, result_list, 10)

    metrics = {"Top1 Track Hit": top1_track_hit, "Top10 Track Hit": top10_track_hit}

    return metrics


def aggregate_track_id_metrics(eval_metrics: list[dict]) -> dict:
    """Calculate the top-1 track hit rate and its symmetric confidence interval."""

    top1_track_hit = np.array([m["Top1 Track Hit"] for m in eval_metrics])
    av_hit_rate_1, ci_1 = compute_proportion_ci(top1_track_hit)
    av_hit_rate_1 = np.round(100 * av_hit_rate_1, 1)
    ci_1 = np.round(100 * ci_1, 1)
    print(f"Top-1 track hit rate: {av_hit_rate_1} ± {ci_1}")

    top10_track_hit = np.array([m["Top10 Track Hit"] for m in eval_metrics])
    av_hit_rate_10, ci_10 = compute_proportion_ci(top10_track_hit)
    av_hit_rate_10 = np.round(100 * av_hit_rate_10, 1)
    ci_10 = np.round(100 * ci_10, 1)
    print(f"Top-10 track hit rate: {av_hit_rate_10} ± {ci_10}")

    metrics = {
        "M-T1HR": av_hit_rate_1,
        "CI-T1HR": ci_1,
        "M-T10HR": av_hit_rate_10,
        "CI-T10HR": ci_10,
    }

    return metrics


def compute_proportion_ci(measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    assert measurements.ndim == 1
    assert set(np.unique(measurements)) <= {0, 1}, "Only 0s and 1s allowed"

    n = len(measurements)
    assert n > 0, "Measurements are empty."

    p = measurements.mean()
    assert p > 0 and p < 1, f"Proportion p must be in (0, 1), got {p}"
    margin = 1.96 * np.sqrt(p * (1 - p) / n)

    return p, margin
