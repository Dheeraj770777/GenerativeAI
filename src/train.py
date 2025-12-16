#!/usr/bin/env python3
"""Train a Random Forest and save the model to `models/`."""
import argparse
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .model import build_model
from .utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train a Random Forest classifier")
    p.add_argument("--dataset", default="iris", choices=["iris", "synthetic"], help="Dataset to use")
    p.add_argument("--output", default="models/model.joblib", help="Output path for saved model")
    p.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    return p.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    model = build_model(n_estimators=args.n_estimators, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
