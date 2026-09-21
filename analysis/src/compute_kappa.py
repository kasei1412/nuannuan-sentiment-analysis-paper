# -*- coding: utf-8 -*-
"""计算人工标注与弱标注的一致性指标（Cohen's Kappa、一致率、混淆矩阵）。

读取 results/tables/annotation_task.csv（需人工填写 human_label 列后运行）。
输出 results/tables/label_quality_report.txt 和 label_confusion_matrix.csv。
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "results" / "tables" / "annotation_task.csv"
REPORT = ROOT / "results" / "tables" / "label_quality_report.txt"
CONFUSION_CSV = ROOT / "results" / "tables" / "label_confusion_matrix.csv"

VALID_LABELS = {"positive", "negative", "neutral"}


def main() -> None:
    df = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            df = pd.read_csv(TASK, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise RuntimeError(f"无法识别 {TASK} 编码")
    # 过滤掉未标注的行
    df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
    labeled = df[df["human_label"].isin(VALID_LABELS)].copy()
    if len(labeled) == 0:
        print("annotation_task.csv 中无有效人工标签（human_label 列为空）")
        return

    y_human = labeled["human_label"].tolist()
    y_weak = labeled["weak_label"].tolist()

    # 一致率
    agree = sum(h == w for h, w in zip(y_human, y_weak))
    agree_rate = agree / len(labeled)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_human, y_weak)

    # 混淆矩阵
    labels_sorted = sorted(VALID_LABELS)
    cm = confusion_matrix(y_human, y_weak, labels=labels_sorted)

    lines = [
        f"标注样本数（已填 human_label）: {len(labeled)}",
        f"一致条数: {agree}",
        f"一致率: {agree_rate:.4f} ({agree_rate*100:.1f}%)",
        f"Cohen's Kappa: {kappa:.4f}",
        "",
        "Kappa 解读: <0.20 极低 | 0.21-0.40 低 | 0.41-0.60 中等 | 0.61-0.80 高 | >0.80 极高",
        "",
        f"混淆矩阵（行=人工标注，列=弱标注）:",
        f"{'':>12}" + "".join(f"{l:>12}" for l in labels_sorted),
    ]
    for i, row_label in enumerate(labels_sorted):
        lines.append(f"{row_label:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(len(labels_sorted))))

    # 不一致清单
    disagree = labeled[labeled["human_label"] != labeled["weak_label"]]
    if len(disagree) > 0:
        lines.append("")
        lines.append(f"不一致样本 ({len(disagree)} 条):")
        for _, r in disagree.iterrows():
            text_preview = str(r["comment"]).replace("\n", " ")[:60]
            lines.append(f"  [{r['game_name']}] 弱={r['weak_label']} 人={r['human_label']} | {text_preview}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:10]))

    # 混淆矩阵 CSV
    with CONFUSION_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["human\\weak"] + labels_sorted)
        for i, rl in enumerate(labels_sorted):
            w.writerow([rl] + list(cm[i]))

    print(f"\n已写出: {REPORT}")
    print(f"已写出: {CONFUSION_CSV}")

    if len(disagree) > 0:
        print(f"\n⚠ 有 {len(disagree)} 条不一致，建议复核后写入 data/dict/label_overrides.csv")


if __name__ == "__main__":
    main()
