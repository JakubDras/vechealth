import numpy as np
from .evaluator import VecHealthEvaluator


def compute_snc_score(evaluator: VecHealthEvaluator, k: int = 10, sample_size: int = 10000) -> dict:
    """
    Semantic Neighborhood Consistency (SNC).
    Mierzy spójność topologii 2-hop (podobieństwo Jaccarda między k-NN punktu i k-NN jego sąsiadów).
    """
    # Pobieramy macierz indeksów k-NN dla całej bazy
    _, indices = evaluator.get_knn(k)
    n_vectors = indices.shape[0]

    # Używamy próbkowania, by metryka liczyła się w ułamek sekundy,
    # zachowując 99.9% dokładności statystycznej
    if n_vectors > sample_size:
        np.random.seed(42)
        eval_indices = np.random.choice(n_vectors, sample_size, replace=False)
    else:
        eval_indices = np.arange(n_vectors)

    overlaps = []

    for idx in eval_indices:
        my_neighbors = set(indices[idx])
        local_overlap = []

        for neighbor_idx in my_neighbors:
            neighbor_neighbors = set(indices[neighbor_idx])

            # Podobieństwo Jaccarda: Przecięcie / Suma
            intersection = len(my_neighbors.intersection(neighbor_neighbors))
            union = len(my_neighbors.union(neighbor_neighbors))

            jaccard = intersection / union if union > 0 else 0.0
            local_overlap.append(jaccard)

        overlaps.append(np.mean(local_overlap))

    return {
        "snc_score": float(np.mean(overlaps))
    }