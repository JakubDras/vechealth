import numpy as np
from .evaluator import VecHealthEvaluator


def compute_outlier_score(evaluator: VecHealthEvaluator, distance_threshold: float = 1.2) -> dict:
    """
    Wykrywa wektory "śmieciowe" (Outliers / Garbage Embeddings).
    W wysokim wymiarze losowy wektor jest ortogonalny do reszty, więc jego dystans
    Euklidesowy do 1-NN dąży do sqrt(2) ~= 1.41.
    """
    # k=1 wystarczy, odcina self-match, daje nam odległość do pierwszego sąsiada
    distances, _ = evaluator.get_knn(k=1)
    closest_distances = distances[:, 0]

    # Frakcja wektorów, których najbliższy sąsiad jest nienaturalnie daleko
    outlier_fraction = np.mean(closest_distances > distance_threshold)

    # Maksymalna odległość w bazie (jako wskaźnik najgorszego outlier'a)
    max_1nn_distance = float(np.max(closest_distances))

    # Odchylenie standardowe sąsiedztw - wzrośnie, bo mamy dwie populacje
    # (prawdziwe dokumenty blisko siebie, outliers daleko)
    std_1nn = float(np.std(closest_distances))

    return {
        "outlier_fraction": float(outlier_fraction),
        "max_1nn_distance": max_1nn_distance,
        "std_1nn_distance": std_1nn
    }