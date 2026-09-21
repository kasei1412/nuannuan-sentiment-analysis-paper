"""一键运行：合并爬取CSV → 预处理 → 情感分类 → LDA。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from build_dataset import build as build_dataset  # noqa: E402
from build_dataset import iter_source_files  # noqa: E402
from lda_topics import run_lda  # noqa: E402
from make_sample_data import main as make_sample  # noqa: E402
from preprocess import preprocess  # noqa: E402
from sentiment_ml import run_sentiment  # noqa: E402


def main() -> None:
    raw = ROOT / "data" / "raw" / "comments.csv"
    raw_all = ROOT / "data" / "raw" / "comments_all.csv"
    sources = iter_source_files()

    print("\n[0/3] 构建数据集")
    if sources:
        build_dataset()
    elif not raw.exists():
        print("未检测到爬取 CSV，生成演示样本……")
        make_sample()
    else:
        print(f"使用已有: {raw}")

    print("\n[1/3] 预处理")
    preprocess(raw, ROOT / "data" / "processed" / "comments_clean.csv", require_label=True)
    if raw_all.exists():
        preprocess(
            raw_all,
            ROOT / "data" / "processed" / "comments_clean_all.csv",
            require_label=False,
        )

    print("\n[2/3] 情感分类 (NB / LR / SVM)")
    run_sentiment()

    print("\n[3/3] LDA 主题建模")
    lda_input = ROOT / "data" / "processed" / "comments_clean_all.csv"
    if not lda_input.exists():
        lda_input = ROOT / "data" / "processed" / "comments_clean.csv"
    run_lda(lda_input)

    print("\n全部完成。请查看 results/tables 与 results/figures。")


if __name__ == "__main__":
    main()
