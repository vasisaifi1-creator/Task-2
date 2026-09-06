"""
cancer_classifier.py
---------------------
Breast Cancer Diagnosis Classifier

Dataset: Wisconsin Breast Cancer Diagnostic Dataset (real, not synthetic).
569 real patient samples. Each row is a fine-needle-aspirate (FNA) biopsy
of a breast mass, with 30 features computed from digitized images of the
cell nuclei (radius, texture, perimeter, area, smoothness, etc. - each as
mean, standard error, and "worst" value).

Source: UCI Machine Learning Repository / Kaggle
(https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI ML Repository.
https://doi.org/10.24432/C5DW2B

Target: diagnosis -> M (Malignant) or B (Benign)

Why this project is a great precision/recall case study:
A false negative here (predicting "Benign" when the tumor is actually
Malignant) is far more dangerous than a false positive. That means recall
on the malignant class matters more than raw accuracy - this script
deliberately walks through that trade-off.

Run:
    python src/cancer_classifier.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "breast_cancer_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

RANDOM_STATE = 42
TARGET = "diagnosis"


# --------------------------------------------------------------------------
# 1. LOAD & CLEAN
# --------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop the stray trailing "Unnamed: 32" column (an artifact of the
    # source CSV's trailing comma) and the "id" column (not predictive).
    drop_cols = [c for c in df.columns if "Unnamed" in c] + ["id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Loaded {len(df)} real patient samples, {df.shape[1] - 1} features")
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nClass distribution:\n", df[TARGET].value_counts())
    print(f"({df[TARGET].value_counts(normalize=True).round(3).to_dict()})")
    return df


def explore_data(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.countplot(x=TARGET, data=df, ax=axes[0, 0], order=["B", "M"])
    axes[0, 0].set_title("Class Balance: Benign vs Malignant")
    axes[0, 0].set_xticklabels(["Benign (357)", "Malignant (212)"])

    sns.boxplot(x=TARGET, y="radius_mean", data=df, ax=axes[0, 1], order=["B", "M"])
    axes[0, 1].set_title("Mean Radius vs Diagnosis")

    sns.boxplot(x=TARGET, y="concavity_mean", data=df, ax=axes[1, 0], order=["B", "M"])
    axes[1, 0].set_title("Mean Concavity vs Diagnosis")

    sns.boxplot(x=TARGET, y="concave points_mean", data=df, ax=axes[1, 1], order=["B", "M"])
    axes[1, 1].set_title("Mean Concave Points vs Diagnosis")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_overview.png", dpi=150)
    plt.close()
    print(f"Saved EDA plots to {OUTPUT_DIR}/eda_overview.png")

    # Correlation heatmap of the "mean" features (most interpretable subset)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[mean_cols].corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation (mean features)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved correlation heatmap to {OUTPUT_DIR}/correlation_heatmap.png")


# --------------------------------------------------------------------------
# 2. SPLIT
# --------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"B": 0, "M": 1})  # 1 = Malignant (positive class)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# --------------------------------------------------------------------------
# 3. TRAIN & COMPARE
# --------------------------------------------------------------------------
def train_and_compare(X_train, y_train):
    """
    Compare models on RECALL for the malignant class, not accuracy.
    In cancer diagnosis, missing a malignant tumor (false negative) is the
    costly mistake - a false positive just means an extra follow-up test.
    """
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        cv_recall = cross_val_score(pipe, X_train, y_train, cv=5, scoring="recall").mean()
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
        results[name] = cv_recall
        print(f"{name:24s} | CV Recall (malignant): {cv_recall:.4f} | CV F1: {cv_f1:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest model by cross-validated recall on malignant class: {best_name}")
    return fitted_pipelines[best_name], best_name, fitted_pipelines


def tune_best_model(X_train, y_train, model_name: str):
    """Light hyperparameter tuning, optimizing for recall."""
    if model_name == "Support Vector Machine":
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=RANDOM_STATE))])
        grid_params = {"classifier__C": [0.1, 1, 10], "classifier__gamma": ["scale", "auto"], "classifier__kernel": ["rbf", "linear"]}
    elif model_name == "Random Forest":
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))])
        grid_params = {"classifier__n_estimators": [200, 300, 400], "classifier__max_depth": [4, 6, 8, None]}
    elif model_name == "Logistic Regression":
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        grid_params = {"classifier__C": [0.01, 0.1, 1, 10]}
    else:
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", KNeighborsClassifier())])
        grid_params = {"classifier__n_neighbors": [3, 5, 7, 9, 11]}

    grid = GridSearchCV(pipe, grid_params, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("\nBest params:", grid.best_params_)
    print("Best CV Recall:", grid.best_score_)
    return grid.best_estimator_


# --------------------------------------------------------------------------
# 4. EVALUATE + THE PRECISION/RECALL TRADE-OFF
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== {model_name} Evaluation ===")
    print("Confusion Matrix (rows=actual, cols=predicted):\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Precision (malignant): {precision:.4f}")
    print(f"Recall (malignant):    {recall:.4f}  <- most critical metric here")
    print(f"F1 Score:               {f1:.4f}")
    print(f"ROC-AUC:                {auc:.4f}")

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Predicted Benign", "Predicted Malignant"],
                yticklabels=["Actual Benign", "Actual Malignant"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # Precision-Recall trade-off curve (the "Strategic Trade-offs" story)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4.5))
    plt.plot(thresholds, prec_curve[:-1], label="Precision", color="#2166AC")
    plt.plot(thresholds, rec_curve[:-1], label="Recall", color="#D6604D")
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default threshold (0.5)")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision vs Recall Trade-off - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/precision_recall_tradeoff.png", dpi=150)
    plt.close()

    print(f"\nSaved confusion matrix, ROC curve, and precision/recall trade-off plots to {OUTPUT_DIR}/")
    return y_pred, y_proba, {"precision": precision, "recall": recall, "f1": f1, "auc": auc}


def demonstrate_threshold_tuning(model, X_test, y_test):
    """
    Shows exactly why 'accuracy' alone is misleading in medical diagnosis.
    Lowering the decision threshold below 0.5 trades some precision for
    higher recall - catching more true malignant cases at the cost of a
    few more false alarms. In cancer screening, that trade is often worth it.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Threshold Tuning: The Precision/Recall Trade-off ===")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'Missed Malignant (FN)':<25}")
    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        y_pred_t = (y_proba >= t).astype(int)
        p = precision_score(y_test, y_pred_t)
        r = recall_score(y_test, y_pred_t)
        fn = ((y_test == 1) & (y_pred_t == 0)).sum()
        print(f"{t:<12}{p:<12.3f}{r:<12.3f}{fn:<25}")
    print("\nLowering the threshold catches more malignant cases (fewer FN),")
    print("at the cost of more false alarms (lower precision). In screening,")
    print("that's usually the right trade to make.")


# --------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------------------------------
def plot_feature_importance(model, model_name, feature_names, X_test=None, y_test=None):
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        order = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances[order], y=np.array(feature_names)[order], color="#B2182B")
        plt.title(f"Top 15 Feature Importances - {model_name}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
        plt.close()
        print(f"Saved feature importance plot to {OUTPUT_DIR}/feature_importance.png")
    elif hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]
        order = np.argsort(np.abs(coefs))[::-1][:15]
        plt.figure(figsize=(8, 6))
        colors = ["#B2182B" if c > 0 else "#2166AC" for c in coefs[order]]
        sns.barplot(x=coefs[order], y=np.array(feature_names)[order], palette=colors)
        plt.title(f"Top 15 Feature Coefficients - {model_name}\n(red = pushes toward Malignant)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
        plt.close()
        print(f"Saved feature coefficient plot to {OUTPUT_DIR}/feature_importance.png")
    elif X_test is not None and y_test is not None:
        # SVM (RBF) and other black-box models: use permutation importance
        # instead - it measures how much performance drops when a feature's
        # values are randomly shuffled, which works for any model type.
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="recall"
        )
        order = np.argsort(result.importances_mean)[::-1][:15]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=result.importances_mean[order], y=np.array(feature_names)[order], color="#B2182B")
        plt.title(f"Top 15 Features by Permutation Importance - {model_name}\n(drop in recall when feature is shuffled)")
        plt.xlabel("Mean Recall Decrease")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
        plt.close()
        print(f"Saved permutation importance plot to {OUTPUT_DIR}/feature_importance.png")
    else:
        print(f"{model_name} does not expose interpretable feature weights.")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_data()
    explore_data(df)

    X_train, X_test, y_train, y_test, feature_names = split_data(df)
    best_model, best_name, all_models = train_and_compare(X_train, y_train)

    print(f"\nRunning GridSearchCV to fine-tune {best_name}...")
    best_model = tune_best_model(X_train, y_train, best_name)

    y_pred, y_proba, metrics = evaluate_model(best_model, X_test, y_test, f"{best_name} (Tuned)")
    demonstrate_threshold_tuning(best_model, X_test, y_test)
    plot_feature_importance(best_model, f"{best_name} (Tuned)", feature_names, X_test, y_test)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
