# -*- coding: utf-8 -*-
"""Event-window analysis: sentiment distribution and topic comparison."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from lda_topics import (
    LDA_GENERIC_STOPWORDS,
    LDA_SYNONYMS,
    distinctive_terms,
    make_topic_docs,
    select_topic_count,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_ALL = ROOT / "data" / "raw" / "comments_all.csv"
CLEAN_ALL = ROOT / "data" / "processed" / "comments_clean_all.csv"
MODEL = ROOT / "results" / "models" / "best_SVM.joblib"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

EVENT_NAME = {"engine_2024": "2024引擎升级", "rebirth_2026": "2026新生版"}
PHASE_NAME = {"pre": "前窗", "post": "后窗", "post_supp": "后窗补充"}
SRC_NAME = {"search": "搜索博文", "official_post_comment": "官方帖评论"}
EVENT_TOPIC_STOPWORDS = {
    "engine_2024": {"闪耀暖暖"},
    "rebirth_2026": {"新生", "新生版", "周年", "七周年", "周年庆", "闪耀暖暖", "优化", "颗星", "点亮"},
}


def keyword_hits(text: str, kws: list[str]) -> int:
    return sum(1 for k in kws if k in text)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_ALL)
    clean = pd.read_csv(CLEAN_ALL)
    model = joblib.load(MODEL)
    clean["pred_label"] = model.predict(clean["tokens"].astype(str))

    # 规整 phase：2024 后窗补充并入后窗
    def norm_phase(event: str, phase: str) -> str:
        if event == "engine_2024" and phase == "post_supp":
            return "post"
        return phase

    clean["phase_norm"] = [norm_phase(e, p) for e, p in zip(clean["event"], clean["phase"])]

    # 1) 情感分布（SVM 二分类 + 弱标注三分类）
    rows = []
    for (event, phase, src), sub in clean[clean["event"] != ""].groupby(["event", "phase_norm", "source_type"]):
        n = len(sub)
        svm_pos = int((sub["pred_label"] == "positive").sum())
        svm_neg = int((sub["pred_label"] == "negative").sum())
        wk_pos = int((sub["label"] == "positive").sum())
        wk_neg = int((sub["label"] == "negative").sum())
        wk_neu = int((sub["label"] == "neutral").sum())
        rows.append({
            "event": event,
            "event_name": EVENT_NAME[event],
            "phase": phase,
            "phase_name": PHASE_NAME[phase],
            "source_type": src,
            "source_name": SRC_NAME[src],
            "n": n,
            "svm_positive": svm_pos,
            "svm_negative": svm_neg,
            "svm_positive_ratio": round(svm_pos / n, 4) if n else 0.0,
            "svm_negative_ratio": round(svm_neg / n, 4) if n else 0.0,
            "weak_positive": wk_pos,
            "weak_negative": wk_neg,
            "weak_neutral": wk_neu,
        })

    sent = pd.DataFrame(rows).sort_values(["event", "phase", "source_type"]).reset_index(drop=True)
    sent.to_csv(TAB / "special_sentiment.csv", index=False, encoding="utf-8-sig")
    print("===== 专项情感分布（清理后可分析文本，SVM 预测 + 弱标注） =====")
    print(sent.to_string(index=False))

    topic_rows = []
    selection_rows = []
    # 事件窗口定义
    windows = [
        ("engine_2024", "pre", "search"),
        ("engine_2024", "post", "search"),          # 含 post_supp
        ("rebirth_2026", "pre", "official_post_comment"),
        ("rebirth_2026", "pre", "search"),
        ("rebirth_2026", "post", "search"),
    ]
    for event, phase, src in windows:
        sub = clean[(clean["event"] == event) & (clean["phase_norm"] == phase) & (clean["source_type"] == src)]
        topic_stopwords = EVENT_TOPIC_STOPWORDS[event] | LDA_GENERIC_STOPWORDS
        docs = make_topic_docs(
            sub["tokens"].tolist(),
            topic_stopwords,
            drop_substrings={"闪耀暖暖", "闪暖", "闪耀", "暖暖"},
            synonym_map=LDA_SYNONYMS,
        )
        if len(docs) < 6:
            print(f"跳过 LDA: {event}-{phase}-{src} 文档过少 ({len(docs)})")
            continue
        model, dictionary, k, scores = select_topic_count(
            docs, max_topics=3, passes=20, coherence_max_docs=120, coherence_method="u_mass"
        )
        for score in scores:
            selection_rows.append(
                {
                    "event": event,
                    "event_name": EVENT_NAME[event],
                    "phase": phase,
                    "phase_name": PHASE_NAME[phase],
                    "source_type": src,
                    "source_name": SRC_NAME[src],
                    "k": score["k"],
                    "coherence_method": "u_mass",
                    "excluded_terms": " ".join(sorted(topic_stopwords)),
                    "coherence_cv": score["coherence_cv"],
                    "selected": int(score["k"] == k),
                }
            )
        for tid in range(model.num_topics):
            terms = distinctive_terms(model, dictionary, tid, topn=8)
            raw_terms = model.show_topic(tid, topn=8)
            topic_rows.append({
                "event": event,
                "event_name": EVENT_NAME[event],
                "phase": phase,
                "phase_name": PHASE_NAME[phase],
                "source_type": src,
                "source_name": SRC_NAME[src],
                "topic_id": tid,
                "keywords": " ".join([w for w, _ in terms]),
                "weights": ";".join([f"{w}:{wt:.4f}" for w, wt in terms]),
                "raw_keywords": " ".join([w for w, _ in raw_terms]),
            })
        print(f"\nLDA[{EVENT_NAME[event]} / {PHASE_NAME[phase]} / {SRC_NAME[src]}] n={len(docs)}")
        for tid in range(model.num_topics):
            print("  ", tid, " ".join(w for w, _ in model.show_topic(tid, topn=8)))

    lda_tab = pd.DataFrame(topic_rows)
    lda_tab.to_csv(TAB / "special_lda_topics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selection_rows).to_csv(
        TAB / "special_lda_model_selection.csv", index=False, encoding="utf-8-sig"
    )

    # 3) 关注点关键词频（前后窗对比，基于清理文本）
    concern_kws = {
        "强制更新/重装": ["强制更新", "重装", "卸载", "安装包", "重新下载"],
        "性能(发热/闪退/卡顿/掉帧)": ["发热", "闪退", "卡顿", "掉帧", "卡死"],
        "内存/存储": ["内存", "存储", "占内存", "空间"],
        "画质/光照/引擎": ["光照", "画质", "引擎", "建模", "渲染", "精致"],
        "期待/激动": ["期待", "等不及", "激动", "好想", "终于"],
        "预约/福利/奖励": ["预约", "福利", "奖励", "礼包", "补偿"],
        "质疑/建议/担忧": ["质疑", "建议", "担心", "能不能", "希望", "优化"],
    }
    freq_rows = []
    for event in ["engine_2024", "rebirth_2026"]:
        for phase in ["pre", "post"]:
            sub = clean[(clean["event"] == event) & (clean["phase_norm"] == phase)]
            texts = sub["comment_clean"].astype(str).tolist()
            n = len(texts)
            for label, kws in concern_kws.items():
                cnt = sum(1 for t in texts if any(k in t for k in kws))
                freq_rows.append({
                    "event": event,
                    "event_name": EVENT_NAME[event],
                    "phase": phase,
                    "phase_name": PHASE_NAME[phase],
                    "concern": label,
                    "n": n,
                    "hits": cnt,
                    "ratio": round(cnt / n, 4) if n else 0.0,
                })
    freq = pd.DataFrame(freq_rows)
    freq.to_csv(TAB / "special_keyword_freq.csv", index=False, encoding="utf-8-sig")

    # 4) 前后窗负面比例差异及 95% 置信区间。
    from math import erfc, sqrt

    def compare_proportions(event: str, pre_src: str) -> dict:
        pre = sent[
            (sent["event"] == event)
            & (sent["phase"] == "pre")
            & (sent["source_type"] == pre_src)
        ].iloc[0]
        post = sent[
            (sent["event"] == event)
            & (sent["phase"] == "post")
            & (sent["source_type"] == "search")
        ].iloc[0]
        p_pre = float(pre["svm_negative_ratio"])
        p_post = float(post["svm_negative_ratio"])
        n_pre, n_post = int(pre["n"]), int(post["n"])
        diff = p_post - p_pre
        se = sqrt(p_pre * (1 - p_pre) / n_pre + p_post * (1 - p_post) / n_post)
        z = diff / se if se else 0.0
        p_value = erfc(abs(z) / sqrt(2))
        return {
            "event": event,
            "event_name": EVENT_NAME[event],
            "pre_source_type": pre_src,
            "pre_n": n_pre,
            "post_n": n_post,
            "pre_negative_ratio": p_pre,
            "post_negative_ratio": p_post,
            "difference_post_minus_pre": round(diff, 4),
            "ci95_low": round(diff - 1.96 * se, 4),
            "ci95_high": round(diff + 1.96 * se, 4),
            "p_value": round(p_value, 4),
        }

    pd.DataFrame(
        [
            compare_proportions("engine_2024", "search"),
            compare_proportions("rebirth_2026", "official_post_comment"),
        ]
    ).to_csv(TAB / "special_sentiment_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n===== 关注点关键词频（占该窗口可分析文本比例） =====")
    pivot = freq.pivot_table(index=["event_name", "phase_name"], columns="concern", values="ratio")
    print(pivot.to_string())

    print("\n完成：专项情感、LDA选择与关键词频结果已写入 results/tables")


if __name__ == "__main__":
    main()
