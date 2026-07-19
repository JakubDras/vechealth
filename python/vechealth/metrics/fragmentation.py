import numpy as np
from .evaluator import VecHealthEvaluator


def compute_fragmentation_score(evaluator: VecHealthEvaluator, k: int = 10) -> dict:
    """
    Neighborhood Dispersion Score (NDS).
    Diagnozuje fragmentację klastrów poprzez badanie globalnej rozrzedzoności sąsiedztw.
    """
    # Zwraca odległości do sąsiadów od 1 do 10 (odcina samego siebie)
    distances, _ = evaluator.get_knn(k)

    # Średni dystans do najbliższego sąsiada (1-NN)
    mean_1nn = np.mean(distances[:, 0])

    # Średni dystans w całym oknie (10-NN)
    mean_knn = np.mean(distances)

    return {
        "mean_1nn_distance": float(mean_1nn),
        "mean_knn_distance": float(mean_knn)
    }