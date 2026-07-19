import numpy as np
import faiss


class VecHealthEvaluator:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.n_vectors = len(vectors)
        self._knn_indices = None
        self._knn_distances = None  # NOWE: Cache dla odległości

    def get_knn(self, k: int = 10):
        """
        Zwraca (distances, indices) dla k najbliższych sąsiadów.
        Odległości są przeliczone z Inner Product na dystans Euklidesowy (L2).
        """
        if self._knn_indices is None or self._knn_indices.shape[1] < k:
            dim = self.vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.vectors)

            sims, indices = index.search(self.vectors, k + 1)

            # Odcinamy pierwszą kolumnę (self-match)
            self._knn_indices = indices[:, 1:k + 1]
            sims = sims[:, 1:k + 1]

            # Przeliczanie Cosine Similarity (IP) na Euclidean Distance
            # Zabezpieczenie przed ujemnymi wartościami (błąd precyzji zmiennoprzecinkowej)
            self._knn_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims))

        return self._knn_distances[:, :k], self._knn_indices[:, :k]