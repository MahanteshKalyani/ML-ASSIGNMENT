"""Streamlit front end for the five saved breast-cancer classifiers.

The uploaded CSV is scored with the model loaded from disk. Nothing is retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
META = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURE_COLUMNS = META["feature_columns"]
TARGET_COLUMN = META["target_column"]
MODEL_FILES = META["model_files"]
BUNDLED_TEST = ROOT / "test_data.csv"
POSITIVE_LABEL = 1


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )
    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Feature columns must be numeric. Found non-numeric or empty cells.")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("The diagnosis column must contain only 0 (benign) and 1 (malignant).")
    return features, target.astype(int)


def score_model(name: str, features: pd.DataFrame, target: pd.Series) -> dict:
    model = load_model(name)
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    malignant_index = list(model.classes_).index(POSITIVE_LABEL)
    malignant_score = probabilities[:, malignant_index]
    labels = [0, 1]
    return {
        "Accuracy": float(accuracy_score(target, predicted)),
        "AUC": float(roc_auc_score(target, malignant_score)),
        "Precision": float(precision_score(target, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "Recall": float(recall_score(target, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "F1": float(f1_score(target, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predicted)),
        "predicted": predicted,
        "report": classification_report(
            target,
            predicted,
            labels=labels,
            target_names=["benign (0)", "malignant (1)"],
            zero_division=0,
        ),
        "matrix": confusion_matrix(target, predicted, labels=labels),
    }


def metric_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": round(result["Accuracy"], 4),
                "AUC": round(result["AUC"], 4),
                "Precision": round(result["Precision"], 4),
                "Recall": round(result["Recall"], 4),
                "F1": round(result["F1"], 4),
                "MCC": round(result["MCC"], 4),
            }
        )
    return pd.DataFrame(rows)


def draw_confusion(matrix, title: str):
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["benign", "malignant"],
        yticklabels=["benign", "malignant"],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def main() -> None:
    st.set_page_config(page_title="Breast Cancer Classifiers", layout="wide")
    st.title("Breast cancer diagnostic classifiers")
    st.caption(
        "Five saved models score an uploaded test CSV. Training already happened; "
        "this page only loads the model and reports metrics."
    )

    with st.sidebar:
        st.header("Test data")
        upload = st.file_uploader("Upload a test CSV", type=["csv"])
        use_bundled = st.checkbox("Use the bundled hold-out test set", value=upload is None)
        st.header("Model")
        selected = st.selectbox("Model selection", list(MODEL_FILES))
        st.markdown(
            "**Target encoding**  \n"
            "1 = malignant  \n"
            "0 = benign"
        )

    if upload is not None and not use_bundled:
        raw = pd.read_csv(upload)
        source_name = upload.name
    else:
        raw = pd.read_csv(BUNDLED_TEST)
        source_name = "test_data.csv (bundled hold-out set)"

    try:
        features, target = prepare_frame(raw)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.write(
        f"Scoring **{len(features)}** rows from `{source_name}` "
        f"({int((target == 1).sum())} malignant, {int((target == 0).sum())} benign)."
    )

    results = {name: score_model(name, features, target) for name in MODEL_FILES}
    chosen = results[selected]

    st.subheader(f"Evaluation metrics — {selected}")
    cards = st.columns(6)
    for card, (label, value) in zip(cards, [(k, chosen[k]) for k in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")]):
        card.metric(label, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        figure = draw_confusion(chosen["matrix"], selected)
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.subheader("Classification report")
        st.code(chosen["report"], language="text")

    st.subheader("All five models on this test file")
    st.dataframe(metric_table(results), width="stretch", hide_index=True)
    st.caption(
        "Precision, recall, and F1 use malignant (1) as the positive class. "
        "AUC uses the predicted probability of malignant."
    )


if __name__ == "__main__":
    main()
