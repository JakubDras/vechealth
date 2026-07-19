import numpy as np
from scipy.stats import skew
from .evaluator import VecHealthEvaluator


def compute_hubness_score(evaluator: VecHealthEvaluator, k: int = 10) -> dict:
    """
    Oblicza zjawisko Hubness ("Czarne Dziury" w przestrzeni).
    Wszystkie operacje to czysty NumPy/SciPy - łatwe do portowania na Rust.
    """
    _, knn = evaluator.get_knn(k)

    # Zliczenie k-occurrences (ile razy dany indeks wystąpił w macierzy sąsiedztwa)
    k_occurrences = np.bincount(knn.flatten(), minlength=evaluator.n_vectors)

    return {
        "hubness_skewness": float(skew(k_occurrences)),
        "orphans_fraction": float(np.mean(k_occurrences == 0)),
        "max_occurrences": int(np.max(k_occurrences))
    }