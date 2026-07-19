import numpy as np
from .evaluator import VecHealthEvaluator


def compute_intrinsic_dimensionality(evaluator: VecHealthEvaluator, k: int = 20) -> dict:
    """
    Estymator MLE (Levina-Bickel) wymiarowości wewnętrznej przestrzeni.
    Używamy k=20, ponieważ estymacja ID potrzebuje nieco szerszego okna sąsiedztwa
    niż klasyczne wyszukiwanie (k=10).
    """
    distances, _ = evaluator.get_knn(k)

    # Zabezpieczenie: jeśli odległość to 0 (duplikaty), dodajemy malutki epsilon,
    # żeby uniknąć dzielenia przez zero w logarytmie
    distances = np.maximum(distances, 1e-10)

    # Tk to ostatnia kolumna (najdalszy sąsiad z k) - kształt (N, 1)
    T_k = distances[:, -1:]
    # Tj to wszystkie poprzednie kolumny - kształt (N, k-1)
    T_j = distances[:, :-1]

    # Równanie Leviny-Bickela
    log_ratio = np.log(T_k / T_j)
    sum_log_ratio = np.sum(log_ratio, axis=1)

    # Wyłapujemy przypadki, gdzie punkty leżą dokładnie w tym samym miejscu
    # (suma logarytmów ~ 0 spowodowałaby nieskończony wymiar)
    valid_mask = sum_log_ratio > 1e-5

    id_per_point = np.zeros(evaluator.n_vectors)
    id_per_point[valid_mask] = (k - 1) / sum_log_ratio[valid_mask]

    return {
        "intrinsic_dim_mean": float(np.mean(id_per_point)),
        "intrinsic_dim_median": float(np.median(id_per_point))
    }