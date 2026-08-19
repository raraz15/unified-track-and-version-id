from typing import Tuple, List, Dict, Any

import numpy as np
from skimage.morphology import flood


def get_bounding_box(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Finds the bounding box of a boolean mask.
    Returns: ((row_min, row_max), (col_min, col_max))
    """

    xs, ys = np.nonzero(mask)

    if xs.size == 0:
        raise ValueError("bounding_box: mask contains no True pixels")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return (x_min, x_max), (y_min, y_max)


def find_basins(
    H: np.ndarray,
    global_pct_threshold: float | None = None,
    universal_dist_threshold: float | None = None,
    max_basins: int | None = None,
    remove_method: str = "basin",
) -> List[Dict[str, Any]]:
    """Extract up to max_basins similar regions of H. H is a distance matrix so
    smaller values means more similar. A similar region is defined by a local minimum
    (hole) in H, and the region around it that is below a certain ridge height. The
    ridge height is defined globally as the global_pct_threshold percentile of all finite
    values in H. Holes that are shallower (i.e., height larger) than the global_height_limit
    are rejected. The global_height_limit is computed as the minimum of the percentile-based
    threshold and the universal_dist_threshold (when provided)."""

    assert H.ndim == 2
    assert (
        global_pct_threshold is not None or universal_dist_threshold is not None
    ), "At least one of global_pct_threshold or universal_dist_threshold must be provided"

    tmp = np.array(H, copy=True)

    if max_basins is None:
        max_basins = int(tmp.size)  # effectively unlimited

    # Only use finite values
    finite_values = tmp[np.isfinite(tmp)]

    # No finite values, early exit
    if finite_values.size == 0:
        return []

    threshold_from_pct = np.inf
    if global_pct_threshold is not None:
        threshold_from_pct = np.quantile(finite_values, global_pct_threshold / 100.0)
    threshold_from_val = np.inf
    if universal_dist_threshold is not None:
        threshold_from_val = universal_dist_threshold
    # The effective ceiling for any basin ridge, works with both or no thresholds
    global_height_limit = min(threshold_from_pct, threshold_from_val)

    # Early exit if the whole matrix is already above the threshold
    if np.all(finite_values > global_height_limit):
        return []

    basins = []
    while len(basins) < max_basins:

        # Find the deepest open hole
        hole_coord = np.unravel_index(np.argmin(tmp), tmp.shape)
        hole_height = tmp[hole_coord]

        # Check if we have run out of valid finite holes
        if not np.isfinite(hole_height):
            break

        # If the hole is not deep enough, exit.
        # Since the holes are sorted by depth, this is ok.
        if hole_height >= global_height_limit:
            break

        # Starting from the hole, flood until global_height_limit
        delta_height = max(global_height_limit - hole_height, 0.0)
        # NOTE: connectivity 2 caused overflow issues
        basin_mask = flood(tmp, hole_coord, connectivity=1, tolerance=delta_height)
        basin_area = int(np.sum(basin_mask))

        # Find the smallest rectangle containing the basin
        (r0, r1), (c0, c1) = get_bounding_box(basin_mask)
        box_area = int((r1 - r0 + 1) * (c1 - c0 + 1))

        # remove the basin and optionally more from the next iteration
        if remove_method == "basin":
            tmp[basin_mask] = np.inf
        elif remove_method == "rectangle":
            tmp[r0 : r1 + 1, c0 : c1 + 1] = np.inf
        elif remove_method == "cross":
            tmp[r0 : r1 + 1] = np.inf
            tmp[:, c0 : c1 + 1] = np.inf
        else:
            raise ValueError(f"Unknown remove_method method: {remove_method}")

        # Record the basin
        basins.append(
            {
                "hole_coordinates": hole_coord,
                "hole_height": float(hole_height),
                "basin_area": basin_area,
                "basin_mask": basin_mask,
                "bounding_box_coordinates": ((r0, r1), (c0, c1)),
                "bounding_box_area": box_area,
            }
        )

    return basins
