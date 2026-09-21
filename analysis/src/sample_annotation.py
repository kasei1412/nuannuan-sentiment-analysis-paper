# -*- coding: utf-8 -*-
"""从清洗后训练集分层抽取样本，生成人工标注任务文件。

分层维度：游戏 × 当前弱标签 × 专项事件
输出 results/tables/annotation_task.csv（含空白列 human_label 供人工填写）。
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "comments_clean.csv"
OUT = ROOT / "results" / "tables" / "annotation_task.csv"
SAMPLE_N = 200
SEED = 42


def main() -> None:
    random.seed(SEED)
    df = pd.read_csv(CLEAN)
    df["stratum"] = df["game"] + "_" + df["label"]

    # 按层比例分配抽样数
    strata_counts = df["stratum"].value_counts()
    total = len(df)
    sample_alloc: dict[str, int] = {}
    remaining = SAMPLE_N
    for stratum, count in strata_counts.items():
        alloc = max(2, round(SAMPLE_N * count / total))
        alloc = min(alloc, count, remaining)
        sample_alloc[stratum] = alloc
        remaining -= alloc

    # 补齐未分配的余额
    if remaining > 0:
        for stratum in sample_alloc:
            extra = min(remaining, len(df[df["stratum"] == stratum]) - sample_alloc[stratum])
            if extra > 0:
                sample_alloc[stratum] += extra
                remaining -= extra
            if remaining <= 0:
                break

    rows = []
    for stratum, n in sample_alloc.items():
        sub = df[df["stratum"] == stratum]
        sampled = sub.sample(n=n, random_state=SEED)
        for _, r in sampled.iterrows():
            rows.append({
                "game": r["game"],
                "game_name": r["game_name"],
                "time": r["time"],
                "comment": r["comment"],
                "weak_label": r["label"],
                "source_file": r["source_file"],
                "human_label": "",  # 空白列供人工填写 positive/negative/neutral
            })

    out_df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已生成 {len(out_df)} 条标注任务 -> {OUT}")
    print("分层分布:")
    print(out_df.groupby(["game", "weak_label"]).size().to_string())


if __name__ == "__main__":
    main()
