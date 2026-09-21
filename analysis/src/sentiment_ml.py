"""TF-IDF + NB / LR / SVM 情感分类实验。"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "comments_clean.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def build_models() -> dict[str, Pipeline]:
    return {
        "NaiveBayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", MultinomialNB()),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LinearSVC(random_state=42)),
            ]
        ),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="positive", zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label="positive", zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label="positive", zero_division=0),
    }


def run_sentiment(data_path: Path = PROCESSED) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    X = df["tokens"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    best_name, best_f1, best_model = None, -1.0, None
    reports = []

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        metrics["Model"] = name
        rows.append(metrics)
        reports.append(f"===== {name} =====\n{classification_report(y_test, pred, zero_division=0)}\n")

        cm = confusion_matrix(y_test, pred, labels=["negative", "positive"])
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["negative", "positive"],
            yticklabels=["negative", "positive"],
        )
        plt.title(f"{name} 混淆矩阵")
        plt.xlabel("预测")
        plt.ylabel("真实")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cm_{name}.png", dpi=160)
        plt.close()

        if metrics["F1"] > best_f1:
            best_name, best_f1, best_model = name, metrics["F1"], model

    result = pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1"]]
    result = result.sort_values("F1", ascending=False).reset_index(drop=True)
    result.to_csv(TABLE_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    (TABLE_DIR / "classification_report.txt").write_text("\n".join(reports), encoding="utf-8")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append(
            {
                "Model": name,
                **{
                    f"{metric}_mean": round(float(scores[f"test_{metric}"].mean()), 4)
                    for metric in scoring
                },
                **{
                    f"{metric}_std": round(float(scores[f"test_{metric}"].std()), 4)
                    for metric in scoring
                },
            }
        )
    pd.DataFrame(cv_rows).sort_values("f1_macro_mean", ascending=False).to_csv(
        TABLE_DIR / "model_cv.csv", index=False, encoding="utf-8-sig"
    )

    if best_model is not None:
        joblib.dump(best_model, MODEL_DIR / f"best_{best_name}.joblib")

    # 全量预测，供代际情绪统计与体验诊断
    full_pred = best_model.predict(X)
    df = df.copy()
    df["pred_label"] = full_pred
    df.to_csv(TABLE_DIR / "sentiment_predictions.csv", index=False, encoding="utf-8-sig")

    game_map = {"jqnn": "奇迹暖暖", "yynn": "闪耀暖暖", "wxnn": "无限暖暖"}
    df["game_name"] = df["game"].map(game_map).fillna(df["game"])
    ratio = (
        df.groupby("game_name")["pred_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    ratio.to_csv(TABLE_DIR / "sentiment_by_game.csv", index=False, encoding="utf-8-sig")

    print("情感分类结果：")
    print(result.to_string(index=False))
    print(f"最优模型: {best_name}")
    return result


if __name__ == "__main__":
    run_sentiment()
