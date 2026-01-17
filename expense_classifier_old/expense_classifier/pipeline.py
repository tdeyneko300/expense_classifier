from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


def _to_str_array(values) -> np.ndarray:
    return np.asarray(["" if _is_missing(v) else str(v) for v in values])


def _to_float_array(values) -> np.ndarray:
    out: list[float] = []
    for v in values:
        if _is_missing(v):
            out.append(0.0)
            continue
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    return np.asarray(out, dtype=float)


def _reshape_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1, 1)


def build_pipeline(model_type: str = "logreg") -> Pipeline:
    text_tfidf = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(_to_str_array, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50_000,
                    lowercase=True,
                ),
            ),
        ]
    )

    merchant_tfidf = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(_to_str_array, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20_000,
                    lowercase=True,
                ),
            ),
        ]
    )

    amount_pipe = Pipeline(
        steps=[
            ("to_float", FunctionTransformer(_to_float_array, validate=False)),
            ("reshape", FunctionTransformer(_reshape_1d, validate=False)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("text", text_tfidf, "text"),
            ("merchant", merchant_tfidf, "merchant"),
            ("amount", amount_pipe, "amount"),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model_type_norm = (model_type or "logreg").strip().lower()
    if model_type_norm in {"logreg", "lr", "logistic", "logistic_regression"}:
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight="balanced",
        )
    elif model_type_norm in {"svm", "linear_svm", "linearsvc"}:
        base = LinearSVC(class_weight="balanced")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}. Use 'logreg' or 'svm'.")

    return Pipeline(
        steps=[
            ("pre", pre),
            ("clf", clf),
        ]
    )


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float

