"""TF-IDF + Logistic Regression reference model.

Auxiliary only - answers "how much of this task is solved by lexical features
alone", nothing more. Not M0, not part of the M0-M4/D0 ladder, not part of the
10-fit budget, never registered in `results/runs.csv` (`src.results` strictly
validates experiment names against `src.checkpoint.VALID_EXPERIMENTS`, and
"TFIDF" is deliberately not on that list - corrupting the official namespace
is exactly what that check exists to prevent). Its result lives in its own
record, `TfidfReferenceResult`, saved separately.

Fit on the frozen training split only, using the same normalization policy Task
3 locked (`src.preprocessing.normalize_text`) - no lowercasing/stopword/stemming
choices invented specifically for this reference. TfidfVectorizer's own internal
tokenization (default `token_pattern`, `lowercase=True`) is its necessary
built-in behavior, not extra project-level cleaning, and is used as-is.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from typing import Any, Union

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import CLASS_TO_ID, LABELS
from src.evaluation import compute_metrics
from src.preprocessing import normalize_text

# Part 4: no config specified elsewhere in the project for this reference, so
# this is the conservative, standard starting point the task itself suggests -
# fixed before fitting, not adjusted after seeing a result.
TFIDF_CONFIG: dict[str, Any] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

# Part 5: explicit, standard, no search. lbfgs is multinomial by default for
# multi-class in this sklearn version and handles sparse input without densifying.
LOGREG_CONFIG: dict[str, Any] = {
    "max_iter": 1000,
    "C": 1.0,
    "solver": "lbfgs",
}


@dataclass
class TfidfReferenceResult:
    """Traceable record of one TF-IDF reference run. Deliberately not
    `RUNS_SCHEMA`-shaped and not written to `results/runs.csv` - see module
    docstring."""
    label: str  # always "TFIDF_REFERENCE" - never M0-M4/D0
    dataset_content_sha256: str
    preprocessing_version: str
    tfidf_config: dict[str, Any]
    logreg_config: dict[str, Any]
    vocabulary_size: int
    macro_f1: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    per_class: dict[str, Any]
    confusion_matrix: list
    val_macro_f1: float
    runtime_seconds: float
    sklearn_version: str
    python_version: str
    git_commit: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TfidfReferenceResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


def fit_tfidf_reference(
    train_texts: pd.Series,
    train_labels: pd.Series,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fits on TRAIN ONLY. `vectorizer.transform(...)` on val/test afterward
    never updates its vocabulary or IDF weights - that is simply what
    `.transform()` (vs `.fit_transform()`) means, verified explicitly in
    `tests/test_tfidf_reference.py`."""
    normalized = train_texts.map(normalize_text)
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(normalized)

    y_train = train_labels.map(CLASS_TO_ID).to_numpy()

    model = LogisticRegression(**LOGREG_CONFIG)
    model.fit(X_train, y_train)
    return vectorizer, model


def transform(vectorizer: TfidfVectorizer, texts: pd.Series) -> "sparse.csr_matrix":
    """Same normalization as fitting, train-fitted vocabulary only. Stays
    sparse - never densified."""
    return vectorizer.transform(texts.map(normalize_text))


def top_features_by_class(
    vectorizer: TfidfVectorizer, model: LogisticRegression, k: int = 12
) -> dict[str, list[str]]:
    """The `k` terms with the largest Logistic Regression coefficient for each
    class - what the reference model is actually keying on, for the Part 11/12
    sanity check (sensible complaint vocabulary vs. a suspicious artifact)."""
    terms = np.asarray(vectorizer.get_feature_names_out())
    out = {}
    for label, idx in CLASS_TO_ID.items():
        coefs = model.coef_[idx]
        top_idx = np.argsort(-coefs)[:k]
        out[label] = terms[top_idx].tolist()
    return out


def build_result(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    X_test,
    y_test: np.ndarray,
    val_macro_f1: float,
    dataset_content_sha256: str,
    preprocessing_version: str,
    runtime_seconds: float,
    git_commit: str,
) -> TfidfReferenceResult:
    """Test evaluation only - call this after the model is already fixed, never
    to pick a configuration."""
    import datetime

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, labels=LABELS)

    return TfidfReferenceResult(
        label="TFIDF_REFERENCE",
        dataset_content_sha256=dataset_content_sha256,
        preprocessing_version=preprocessing_version,
        tfidf_config={k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_CONFIG.items()},
        logreg_config=LOGREG_CONFIG,
        vocabulary_size=len(vectorizer.vocabulary_),
        macro_f1=metrics["macro_f1"],
        accuracy=metrics["accuracy"],
        macro_precision=metrics["macro_precision"],
        macro_recall=metrics["macro_recall"],
        per_class=metrics["per_class"],
        confusion_matrix=metrics["confusion_matrix"],
        val_macro_f1=val_macro_f1,
        runtime_seconds=runtime_seconds,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        git_commit=git_commit,
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
