import numpy as np
import faiss


def compute_qmas_score(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> dict:
    """
    Query Manifold Alignment Score (QMAS).
    Mierzy, jak dobrze wektory zapytań pokrywają się z przestrzenią dokumentów.
    Zwraca średni dystans od zapytania do najbliższych dokumentów.
    """
    dim = doc_vectors.shape[1]

    # Budujemy indeks dla dokumentów (używamy FlatIP dla Cosine Similarity)
    # Zakładamy, że wektory są znormalizowane na wejściu (L2 norm = 1.0)
    index = faiss.IndexFlatIP(dim)
    index.add(doc_vectors)

    # Szukamy k najbliższych dokumentów dla KAŻDEGO zapytania
    similarities, _ = index.search(query_vectors, k)

    # Przekształcamy podobieństwo cosinusowe na odległość (Dystans Cosinusowy = 1 - Cosine)
    # Odcinamy ewentualne błędy precyzji float32 (< 0)
    distances = np.clip(1.0 - similarities, 0.0, 2.0)

    # 1. Średnia odległość do pierwszego trafienia (najbliższy dokument dla zapytania)
    qmas_mean_1nn = float(np.mean(distances[:, 0]))

    # 2. Średnia odległość do całego otoczenia (Top-K)
    qmas_mean_knn = float(np.mean(distances))

    # 3. Odsetek zapytań "osieroconych" (które nie mają żadnego dokumentu w promieniu np. 0.3)
    # Promień 0.3 dla dystansu cosinusowego to odpowiednik cos < 0.70
    orphaned_queries = float(np.mean(distances[:, 0] > 0.3))

    return {
        "qmas_mean_1nn": qmas_mean_1nn,
        "qmas_mean_knn": qmas_mean_knn,
        "orphaned_queries_fraction": orphaned_queries
    }