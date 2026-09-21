# -*- coding: utf-8 -*-
"""论文深度分析：主题×情感联合、词典基线对比、特征判别词、错误分析、6 窗口趋势、主题命名。

输出（results/tables 与 results/figures）：
  topic_sentiment_matrix.csv     主题×情感联合矩阵（每主题负面占比）
  topic_sentiment_heatmap.png    主题×情感热力图
  baseline_comparison.csv        词典基线 vs SVM（人工标注子集 189 条）
  feature_weights_top.csv        LR/SVM 正负判别词 Top-20
  error_analysis.csv             测试集误分类样本
  sentiment_by_window.csv        6 个主窗口负面占比
  sentiment_trend.png            6 窗口情绪趋势折线图
  lda_topic_names.csv            主题命名模板（需人工审定）
"""
from __future__ import annotations

import csv
import pathlib
import pickle
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SRC = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from lda_topics import GAME_NAME, make_topic_docs  # noqa: E402
from preprocess import clean_text, load_resources, tokenize  # noqa: E402
from sentiment_rules import enhanced_weak_label  # noqa: E402

TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
CLEAN_ALL = ROOT / "data" / "processed" / "comments_clean_all.csv"
ANNOT = ROOT / "results" / "tables" / "annotation_task.csv"

# 6 个主对比窗口 -> (显示名, 阶段)
MAIN_WINDOWS = [
    ("qjn180805.xlsx", "奇迹 2018-08", "2D"),
    ("qjn191030.csv", "奇迹 2019-10", "2D"),
    ("qjnn191031.xlsx", "奇迹 2019-10", "2D"),
    ("synn191031.xlsx", "闪耀 2019-10", "3D"),
    ("synn210615.xlsx", "闪耀 2021-06", "3D"),
    ("wxnn241208.xlsx", "无限 2024-12", "开放世界"),
    ("wxnn260804.xlsx", "无限 2026-08", "开放世界"),
    ("wxnn260805.xlsx", "无限 2026-08", "开放世界"),
]
WINDOW_GROUPS = [
    (["qjn180805.xlsx"], "奇迹暖暖 2018-08", "2D"),
    (["qjn191030.csv", "qjnn191031.xlsx"], "奇迹暖暖 2019-10", "2D"),
    (["synn191031.xlsx"], "闪耀暖暖 2019-10", "3D"),
    (["synn210615.xlsx"], "闪耀暖暖 2021-06", "3D"),
    (["wxnn241208.xlsx"], "无限暖暖 2024-12", "开放世界"),
    (["wxnn260804.xlsx", "wxnn260805.xlsx"], "无限暖暖 2026-08", "开放世界"),
]


def fmt_ratio(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


# ---------------------------------------------------------------------------
# A. 主题 × 情感联合
# ---------------------------------------------------------------------------
def topic_sentiment_joint() -> pd.DataFrame:
    clean = pd.read_csv(CLEAN_ALL)
    model_svm = joblib.load(MODEL_DIR / "best_SVM.joblib")
    clean["pred"] = model_svm.predict(clean["tokens"].astype(str))

    rows = []
    for game in ["jqnn", "wxnn", "yynn"]:
        with open(MODEL_DIR / f"lda_{game}.pkl", "rb") as fh:
            art = pickle.load(fh)
        lda_model, dictionary = art["model"], art["dictionary"]
        sub = clean[clean["game"] == game].copy()
        if sub.empty:
            continue
        docs = make_topic_docs(
            sub["tokens"].tolist(),
            art["topic_stopwords"],
            drop_substrings=art["drop_substrings"],
            synonym_map=art["synonym_map"],
        )
        topic_ids = []
        for doc in docs:
            bow = dictionary.doc2bow(doc)
            if not bow:
                topic_ids.append(-1)
                continue
            dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
            topic_ids.append(max(dist, key=lambda t: t[1])[0])
        sub = sub.iloc[: len(topic_ids)].copy()
        sub["topic_id"] = topic_ids
        sub = sub[sub["topic_id"] >= 0]
        for tid, grp in sub.groupby("topic_id"):
            n = len(grp)
            neg = int((grp["pred"] == "negative").sum())
            pos = int((grp["pred"] == "positive").sum())
            keywords = " ".join(
                [w for w, _ in lda_model.show_topic(int(tid), topn=8)]
            )
            rows.append(
                {
                    "game": game,
                    "game_name": GAME_NAME[game],
                    "topic_id": int(tid),
                    "keywords": keywords,
                    "n_docs": n,
                    "negative": neg,
                    "positive": pos,
                    "neg_ratio": fmt_ratio(neg / n) if n else 0.0,
                    "pos_ratio": fmt_ratio(pos / n) if n else 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "topic_sentiment_matrix.csv", index=False, encoding="utf-8-sig")

    # 热力图：每游戏一列子图
    games = df["game"].unique()
    fig, axes = plt.subplots(1, len(games), figsize=(4.5 * len(games), 4.0), squeeze=False)
    for ax, game in zip(axes[0], games):
        d = df[df["game"] == game].sort_values("topic_id")
        labels = [f"T{int(t)}\n{kw[:12]}…" for t, kw in zip(d["topic_id"], d["keywords"])]
        matrix = d[["neg_ratio", "pos_ratio"]].values
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            xticklabels=["负面占比", "正面占比"],
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(GAME_NAME[game])
    fig.suptitle("各主题正/负面情绪占比")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "topic_sentiment_heatmap.png", dpi=160)
    plt.close(fig)
    print(df.to_string(index=False))
    print(f"-> {TABLE_DIR / 'topic_sentiment_matrix.csv'}")
    return df


# ---------------------------------------------------------------------------
# B. 词典基线 vs SVM（人工标注子集）
# ---------------------------------------------------------------------------
def baseline_comparison() -> pd.DataFrame:
    annot = pd.read_csv(ANNOT, encoding="gb18030")
    human = annot[annot["human_label"].isin(["positive", "negative"])].copy()
    stopwords = load_resources()
    model_svm = joblib.load(MODEL_DIR / "best_SVM.joblib")

    human["dict_label"] = human["comment"].map(lambda c: enhanced_weak_label(c)[0])
    human["svm_tokens"] = human["comment"].map(lambda c: tokenize(clean_text(c), stopwords))
    human["svm_label"] = model_svm.predict(human["svm_tokens"].astype(str))
    y = human["human_label"].astype(str)

    def metrics(pred) -> dict:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "n": len(human),
            "Accuracy": round(float(accuracy_score(y, pred)), 4),
            "Precision": round(float(precision_score(y, pred, pos_label="positive")), 4),
            "Recall": round(float(recall_score(y, pred, pos_label="positive")), 4),
            "F1": round(float(f1_score(y, pred, pos_label="positive")), 4),
        }

    rows = [
        {"method": "词典规则基线", **metrics(human["dict_label"])},
        {"method": "SVM（TF-IDF）", **metrics(human["svm_label"])},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(
        f"-> {TABLE_DIR / 'baseline_comparison.csv'}（人工标注正负样本 {len(human)} 条；"
        f"词典预测 neutral {int((human['dict_label'] == 'neutral').sum())} 条）"
    )
    return df


# ---------------------------------------------------------------------------
# C. 特征判别词（LR / SVM 权重 Top）
# ---------------------------------------------------------------------------
def feature_weights() -> pd.DataFrame:
    model_svm = joblib.load(MODEL_DIR / "best_SVM.joblib")
    tfidf = model_svm.named_steps["tfidf"]
    svm = model_svm.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    rows = []
    for method, coef in [
        ("SVM", svm.coef_[0]),
        ("LR", None),
    ]:
        if coef is None:
            continue
        order = coef.argsort()
        pos = [(feature_names[i], float(coef[i])) for i in order[::-1][:20]]
        neg = [(feature_names[i], float(coef[i])) for i in order[:20]]
        for rank, (w, v) in enumerate(pos, 1):
            rows.append({"model": method, "polarity": "正面判别", "rank": rank, "word": w, "weight": round(v, 4)})
        for rank, (w, v) in enumerate(neg, 1):
            rows.append({"model": method, "polarity": "负面判别", "rank": rank, "word": w, "weight": round(v, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "feature_weights_top.csv", index=False, encoding="utf-8-sig")
    print(df[df["model"] == "SVM"].to_string(index=False))
    print(f"-> {TABLE_DIR / 'feature_weights_top.csv'}")
    return df


# ---------------------------------------------------------------------------
# D. 错误分析（测试集误分类样本）
# ---------------------------------------------------------------------------
def error_analysis() -> pd.DataFrame:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ROOT / "data/processed/comments_clean.csv")
    X = df["tokens"].astype(str)
    y = df["label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_svm = joblib.load(MODEL_DIR / "best_SVM.joblib")
    pred = model_svm.predict(X_test)

    test = df.loc[X_test.index].copy()
    test["pred_label"] = pred
    err = test[test["label"] != test["pred_label"]].copy()
    err = err.sort_values("time")
    out = err[["game_name", "time", "comment", "label", "pred_label"]]
    out.to_csv(TABLE_DIR / "error_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"测试集 {len(test)} 条，误分类 {len(err)} 条（{len(err) / len(test):.1%}）")
    print(f"误分类方向: {err.groupby(['label', 'pred_label']).size().to_dict()}")
    print(f"-> {TABLE_DIR / 'error_analysis.csv'}（样本明细见 csv，含原文）")
    return out


# ---------------------------------------------------------------------------
# E. 6 窗口情绪趋势
# ---------------------------------------------------------------------------
def sentiment_trend() -> pd.DataFrame:
    clean = pd.read_csv(CLEAN_ALL)
    model_svm = joblib.load(MODEL_DIR / "best_SVM.joblib")
    clean["pred"] = model_svm.predict(clean["tokens"].astype(str))

    rows = []
    for files, label, stage in WINDOW_GROUPS:
        sub = clean[clean["source_file"].isin(files)]
        n = len(sub)
        neg = int((sub["pred"] == "negative").sum())
        rows.append(
            {
                "window": label,
                "stage": stage,
                "n": n,
                "negative": neg,
                "neg_ratio": fmt_ratio(neg / n) if n else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "sentiment_by_window.csv", index=False, encoding="utf-8-sig")

    x = range(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(x), df["neg_ratio"], marker="o", linewidth=2, color="#c0392b")
    ax.plot(list(x), df["neg_ratio"], ls="", marker="o", color="#c0392b")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.annotate(
            f"{r['neg_ratio']:.1%}\n(n={r['n']})",
            (i, r["neg_ratio"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["window"], rotation=20, ha="right")
    ax.set_ylabel("负面占比（SVM 预测）")
    ax.set_title("六主窗口玩家评论负面情绪占比趋势")
    ax.grid(alpha=0.3)
    # 阶段标注
    stage_bound = []
    prev = None
    for i, s in enumerate(df["stage"]):
        if s != prev:
            stage_bound.append((i, s))
            prev = s
    for i, s in stage_bound:
        ax.axvline(i - 0.5, color="gray", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sentiment_trend.png", dpi=160)
    plt.close(fig)
    print(df.to_string(index=False))
    print(f"-> {TABLE_DIR / 'sentiment_by_window.csv'} / {FIG_DIR / 'sentiment_trend.png'}")
    return df


# ---------------------------------------------------------------------------
# F. 主题命名模板
# ---------------------------------------------------------------------------
def topic_names_template() -> pd.DataFrame:
    topics = pd.read_csv(TABLE_DIR / "lda_topics.csv")
    rows = []
    for _, r in topics.iterrows():
        rows.append(
            {
                "game": r["game"],
                "game_name": r["game_name"],
                "topic_id": r["topic_id"],
                "keywords": r["keywords"],
                "candidate_name": "",
                "note": "",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "lda_topic_names.csv", index=False, encoding="utf-8-sig")
    print(f"-> {TABLE_DIR / 'lda_topic_names.csv'}（待人工审定 candidate_name）")
    return df


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("===== A. 主题×情感联合 =====")
    topic_sentiment_joint()
    print("\n===== B. 词典基线对比 =====")
    baseline_comparison()
    print("\n===== C. 特征判别词 =====")
    feature_weights()
    print("\n===== D. 错误分析 =====")
    error_analysis()
    print("\n===== E. 6 窗口趋势 =====")
    sentiment_trend()
    print("\n===== F. 主题命名模板 =====")
    topic_names_template()
    print("\n全部完成。")


if __name__ == "__main__":
    main()
