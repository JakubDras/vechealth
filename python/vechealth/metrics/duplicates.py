import numpy as np
from .evaluator import VecHealthEvaluator


def compute_ndds_score(evaluator: VecHealthEvaluator, k: int = 1, epsilon: float = 0.05) -> dict:
    """
    Near-Duplicate Density Score (NDDS).
    """
    # Wystarczy nam k=1, bo evaluator sam z siebie odrzuca self-match (odległość 0.0)
    distances, _ = evaluator.get_knn(k)

    # Indeks 0 to pierwszy faktyczny sąsiad
    closest_neighbor_distances = distances[:, 0]

    ndds_fraction = np.mean(closest_neighbor_distances < epsilon)
    mean_1nn_dist = float(np.mean(closest_neighbor_distances))
    min_dist = float(np.min(closest_neighbor_distances))

    return {
        "ndds_fraction": float(ndds_fraction),
        "mean_1nn_distance": mean_1nn_dist,
        "min_distance_global": min_dist
    }