"""Train the five required classifiers and save them for the Streamlit app.

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI, distributed with scikit-learn.
The positive class is malignant (recoded to 1). Metrics are computed on a
stratified 20% hold-out set. Models are not refit inside the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = 1


def build_models() -> dict[str, object]:
    """Same five families required by the assignment, each with a fixed seed."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        SVC(kernel="rbf", random_state=RANDOM_STATE),
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def positive_scores(model, x_test) -> object:
    """Probability of the malignant class, used for AUC."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        classes = list(model.classes_)
        return proba[:, classes.index(POSITIVE_LABEL)]
    scores = model.decision_function(x_test)
    return scores


def evaluate(model, x_test, y_test) -> dict:
    predicted = model.predict(x_test)
    scores = positive_scores(model, x_test)
    labels = [0, 1]
    matrix = confusion_matrix(y_test, predicted, labels=labels)
    return {
        "Accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "AUC": round(float(roc_auc_score(y_test, scores)), 4),
        "Precision": round(
            float(precision_score(y_test, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
            4,
        ),
        "Recall": round(
            float(recall_score(y_test, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
            4,
        ),
        "F1": round(
            float(f1_score(y_test, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
            4,
        ),
        "MCC": round(float(matthews_corrcoef(y_test, predicted)), 4),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    bunch = load_breast_cancer()
    # sklearn encodes malignant as 0. Recode so 1 means malignant.
    features = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    target = pd.Series((bunch.target == 0).astype(int), name="diagnosis")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    file_names = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        joblib.dump(model, MODEL_DIR / file_names[name])
        metrics[name] = evaluate(model, x_test, y_test)
        print(name, {k: v for k, v in metrics[name].items() if k != "confusion_matrix"})

    test_frame = x_test.copy()
    test_frame["diagnosis"] = y_test.to_numpy()
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    meta = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository, via sklearn.datasets.load_breast_cancer",
        "n_instances": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "feature_columns": list(features.columns),
        "target_column": "diagnosis",
        "positive_class": "malignant = 1",
        "negative_class": "benign = 0",
        "class_counts": {
            "malignant": int((target == 1).sum()),
            "benign": int((target == 0).sum()),
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
        },
        "model_files": file_names,
        "metrics": metrics,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved test set ({len(test_frame)} rows) and {len(file_names)} models.")


if __name__ == "__main__":
    main()
