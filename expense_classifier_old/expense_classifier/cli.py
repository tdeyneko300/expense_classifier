from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from expense_classifier.io import load_model, save_model
from expense_classifier.pipeline import build_pipeline


REQUIRED_COLUMNS = ("text", "merchant", "amount", "category")
UNLABELED_COLUMNS = ("text", "merchant", "amount")


def _read_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}. Expected: {list(REQUIRED_COLUMNS)}"
        )

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["merchant"] = df["merchant"].fillna("").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
    df["category"] = df["category"].astype(str)

    df = df[df["category"].str.len() > 0]
    if df.empty:
        raise ValueError("Dataset has no labeled rows after cleaning.")

    return df


def _read_unlabeled_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in UNLABELED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}. Expected: {list(UNLABELED_COLUMNS)}"
        )

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["merchant"] = df["merchant"].fillna("").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype(float)
    return df


def _safe_train_test_split(df: pd.DataFrame, test_size: float, random_state: int):
    y = df["category"]
    counts = y.value_counts(dropna=False)
    can_stratify = bool((counts >= 2).all() and len(counts) >= 2)

    return train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )


def cmd_train(args: argparse.Namespace) -> int:
    df = _read_dataset(args.data)

    x_train, x_test, y_train, y_test = _safe_train_test_split(
        df, test_size=args.test_size, random_state=args.random_state
    )

    model = build_pipeline(model_type=args.model_type)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, digits=3)
    print(report)

    save_model(model, args.out)
    print(f"Saved model to: {args.out}")
    return 0


def _predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # binary case -> 2 columns
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    raise TypeError("Model does not support probability/confidence output.")


def cmd_classify(args: argparse.Namespace) -> int:
    model = load_model(args.model)

    x = pd.DataFrame(
        [
            {
                "text": args.text or "",
                "merchant": args.merchant or "",
                "amount": float(args.amount) if args.amount is not None else 0.0,
            }
        ]
    )

    probs = _predict_proba(model, x)[0]

    classes = getattr(model, "classes_", None)
    if classes is None:
        # for Pipeline, classes_ lives on final step
        classes = getattr(getattr(model, "named_steps", {}).get("clf", None), "classes_", None)
    if classes is None:
        raise TypeError("Could not determine class labels (classes_ missing).")

    classes = np.asarray(classes)
    best_idx = int(np.argmax(probs))
    category = str(classes[best_idx])
    confidence = float(probs[best_idx])

    if args.json:
        payload = {
            "category": category,
            "confidence": confidence,
        }
        if args.topk and args.topk > 1:
            top_idx = np.argsort(-probs)[: args.topk]
            payload["topk"] = [
                {"category": str(classes[i]), "confidence": float(probs[i])}
                for i in top_idx
            ]
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(f"category={category} confidence={confidence:.4f}")
    if args.topk and args.topk > 1:
        top_idx = np.argsort(-probs)[: args.topk]
        for i in top_idx:
            print(f"  {classes[i]}\t{float(probs[i]):.4f}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    df = _read_dataset(args.data)

    x = df[["text", "merchant", "amount"]]
    y_true = df["category"]

    y_pred = model.predict(x)
    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy={acc:.4f}")
    print(classification_report(y_true, y_pred, digits=3))
    return 0


def cmd_predict_csv(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    df = _read_unlabeled_dataset(args.data)

    x = df[["text", "merchant", "amount"]]
    probs = _predict_proba(model, x)

    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(getattr(model, "named_steps", {}).get("clf", None), "classes_", None)
    if classes is None:
        raise TypeError("Could not determine class labels (classes_ missing).")
    classes = np.asarray(classes)

    topk = int(args.topk) if args.topk is not None else 1
    topk = max(1, min(topk, len(classes)))

    # Top-1
    best_idx = np.argmax(probs, axis=1)
    df_out = df.copy()
    df_out["predicted_category"] = classes[best_idx].astype(str)
    df_out["confidence"] = probs[np.arange(len(df_out)), best_idx].astype(float)

    # Optional top-k columns
    if topk > 1:
        order = np.argsort(-probs, axis=1)[:, :topk]
        for k in range(topk):
            idx_k = order[:, k]
            df_out[f"top{k+1}_category"] = classes[idx_k].astype(str)
            df_out[f"top{k+1}_confidence"] = probs[np.arange(len(df_out)), idx_k].astype(float)

    out_path = Path(args.out)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="expense-classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model from labeled CSV")
    p_train.add_argument("--data", required=True, help="Path to labeled CSV")
    p_train.add_argument("--out", required=True, help="Path to output .joblib")
    p_train.add_argument(
        "--model-type",
        default="logreg",
        choices=["logreg", "svm"],
        help="Classifier type",
    )
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_clf = sub.add_parser("classify", help="Predict category for one transaction")
    p_clf.add_argument("--model", required=True, help="Path to .joblib")
    p_clf.add_argument("--text", required=True, help="Payment purpose / description")
    p_clf.add_argument("--merchant", default="", help="Merchant name (optional)")
    p_clf.add_argument("--amount", type=float, default=0.0, help="Amount (optional)")
    p_clf.add_argument("--topk", type=int, default=1, help="Show top-k classes")
    p_clf.add_argument("--json", action="store_true", help="Output JSON")
    p_clf.set_defaults(func=cmd_classify)

    p_eval = sub.add_parser("evaluate", help="Evaluate model on labeled CSV")
    p_eval.add_argument("--model", required=True, help="Path to .joblib")
    p_eval.add_argument("--data", required=True, help="Path to labeled CSV")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict-csv", help="Predict categories for an unlabeled CSV")
    p_pred.add_argument("--model", required=True, help="Path to .joblib")
    p_pred.add_argument("--data", required=True, help="Path to CSV with text/merchant/amount")
    p_pred.add_argument("--out", required=True, help="Path to output CSV")
    p_pred.add_argument("--topk", type=int, default=1, help="Also write top-k predictions")
    p_pred.set_defaults(func=cmd_predict_csv)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
