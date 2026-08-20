"""Near-duplicate clustering used to group the train/val/test split.

Exact deduplication is not enough. Lightly edited template letters - mostly
credit-repair and debt-validation boilerplate - survive it, and when one lands in
train and its twin in test the model scores the twin from memory.

Threshold choice. Holding a TF-IDF reference model fixed and slicing the test set
by each document's highest cosine similarity to any training document, against a
baseline of formulaic documents that have no training twin at all (Macro-F1
0.837):

    similarity band   share of test   Macro-F1   vs baseline
    0.80 - 0.90            8.5%        0.928       +0.091
    0.70 - 0.80            1.7%        0.923       +0.086
    0.60 - 0.70            1.8%        0.870       +0.034
    0.50 - 0.60            4.9%        0.856       +0.019
    below 0.50            83.1%        0.849       +0.012

Inflation is flat and large above 0.70 and collapses immediately below it, so
0.70 is where clusters are cut. The residual 0.60-0.70 band is left grouped-out
and reported as a known limitation rather than chased further: below that point
cosine similarity mostly reflects shared product vocabulary, and grouping on it
would start folding the label into the split.

Note the baseline is *lower* than the bulk of the test set (0.837 vs 0.849).
Formulaic complaints are intrinsically harder, not easier, which is what rules
out "template text is just easy to classify" as the explanation for the gap.

Clusters are connected components under the threshold, so membership is
transitive: A joins B's cluster if they are similar to each other, even when A
and the rest of B's cluster are not. That is the conservative direction - it
over-groups rather than letting a near-twin slip across the split boundary. At
this threshold the largest component is ~700 rows, so transitive chaining does
not collapse the corpus into one giant group.
"""

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CLUSTER_THRESHOLD = 0.70


def _build_vectorizer() -> TfidfVectorizer:
    """Same configuration as the Stage 1 audit, so the near-duplicate rates measured
    there and the clusters grouped here describe the same thing."""
    return TfidfVectorizer(
        min_df=3, max_features=60000, sublinear_tf=True, strip_accents="unicode"
    )


def near_duplicate_clusters(
    texts, threshold: float = CLUSTER_THRESHOLD, chunk_size: int = 500
) -> np.ndarray:
    """Connected-component id per document under TF-IDF cosine >= threshold.

    Documents with no near-duplicate get a cluster of their own, so the result is
    always a complete partition and can be passed straight to a grouped split.
    """
    matrix = _build_vectorizer().fit_transform(texts).astype(np.float32)
    n = matrix.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    # Only the upper triangle: pairs are symmetric and self-similarity is 1.0.
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = (matrix[start:stop] @ matrix[start:].T).toarray()
        rows, cols = np.nonzero(block >= threshold)
        for r, c in zip(rows + start, cols + start):
            if r == c:
                continue
            root_r, root_c = find(int(r)), find(int(c))
            if root_r != root_c:
                parent[max(root_r, root_c)] = min(root_r, root_c)

    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return cluster_ids.astype(np.int64)


def save_clusters(clusters: np.ndarray, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, clusters)


def load_clusters(path: Path, expected_rows: int) -> np.ndarray:
    """Clustering the full corpus takes ~10 minutes, so it is computed once and
    frozen alongside the split indices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run python -m scripts.build_splits to create the frozen split."
        )
    clusters = np.load(path)
    if len(clusters) != expected_rows:
        raise ValueError(
            f"{path} holds {len(clusters)} cluster ids but the dataset has {expected_rows} rows"
        )
    return clusters
