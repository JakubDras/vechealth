import numpy as np
from .evaluator import VecHealthEvaluator


def compute_anisotropy_score(evaluator: VecHealthEvaluator) -> dict:
    """
    Wykrywa anizotropię (efekt stożka / zapaść kierunkową).
    Wykorzystuje twierdzenie o wektorze średnim oraz spektrum macierzy kowariancji.
    """
    vectors = evaluator.vectors
    n_vectors, dim = vectors.shape

    # 1. Wskaźnik Ethayarajha (Norma wektora średniego)
    # Odpowiada średniemu cosinusowi między wszystkimi parami w bazie.
    mean_vec = np.mean(vectors, axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))

    # 2. Analiza macierzy kowariancji (Eigen-decomposition)
    # Centrujemy wektory
    centered = vectors - mean_vec

    # Szybkie liczenie kowariancji (X^T * X) / (N - 1)
    # Rozmiar to D x D (np. 4096 x 4096), zużywa ułamek pamięci, liczy się w sekundy
    cov_matrix = (centered.T @ centered) / (n_vectors - 1)

    # Wyciągamy wartości własne (eigh jest silnie zoptymalizowane dla macierzy symetrycznych)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    # Sortujemy malejąco
    eigenvalues = eigenvalues[::-1]

    # Zabezpieczenie przed epsilonami < 0 z błędów numerycznych float32
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total_variance = np.sum(eigenvalues)

    if total_variance > 0:
        top1_ratio = eigenvalues[0] / total_variance
        top10_ratio = np.sum(eigenvalues[:10]) / total_variance
    else:
        top1_ratio = 0.0
        top10_ratio = 0.0

    return {
        "mean_vector_norm": mean_norm,
        "top1_variance_ratio": float(top1_ratio),
        "top10_variance_ratio": float(top10_ratio)
    }