# -*- coding: utf-8 -*-
"""应用人工筛选结果（domain_term_candidates.csv 的 decision 列）。

decision 含义：
  0 -> 停用词：写入 data/dict/lda_stopwords_extra.txt（仅 LDA 场景剔除，
       不影响情感分类特征）；
  1 -> 领域词：追加到 data/dict/game_terms.txt；
  2 -> 分词待处理：满足规则的自动加入词典修复切分，其余写入
       results/tables/pending_segmentation_review.csv 待复核。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "tables" / "domain_term_candidates.csv"
DICT_PATH = ROOT / "data" / "dict" / "game_terms.txt"
STOP_PATH = ROOT / "data" / "dict" / "lda_stopwords_extra.txt"
PENDING_PATH = ROOT / "results" / "tables" / "pending_segmentation_review.csv"
REPORT_PATH = ROOT / "results" / "tables" / "term_decision_summary.txt"

BOUNDARY_EXCLUDE = set(
    "的了是在是我有和就都也都还这么那你不没吧呢啊呀哦么什很被把让给从到与及或而等着过得之上下里外前后完起"
)
# 词内部不允许出现虚词字（防止“闪耀的暖暖”“年老游”等跨词碎片入库）
INTERNAL_EXCLUDE = set("的了是在不没这那有就都也很它她他你我")
CJK_RE = re.compile(r"^[\u4e00-\u9fa5]+$")
START_FUNC_RE = re.compile(r"^(想|要|会|能|有|没|不|是|在|就|还|又|也|才|刚|很|挺|太|好|真|谁|什|怎|哪|这|那|玩|进|来|去|看|说|问|知|道)")
END_FUNC_RE = re.compile(r"(了|吗|呢|吧|呀|哦|的|之|者|呗|嘛|人|们|一个|一下|一大)$")
AUTO_MIN_DOC_FREQ = 12

# 明确的游戏术语，即使频率较低或命中函数词规则也强制入库（仅限清单中实际出现的）
FORCE_ADD = {
    "四鬼", "幻阁", "扭蛋", "奇想星", "收集度", "粉钻", "攒钻", "兑换码",
    "氪条", "暖五", "大世界", "主界面", "滤镜光照", "引擎更新", "苏暖暖",
    "大喵", "掉率", "星光币", "迷海",
}


def read_candidates() -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(CSV_PATH, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("无法识别 domain_term_candidates.csv 编码")


def load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    df = read_candidates()
    decision = df["decision"].fillna("").astype(str).str.strip()
    existing_dict = load_lines(DICT_PATH)

    stop_words = sorted(set(df.loc[decision == "0", "candidate"].astype(str)))
    dict_words_raw = [str(w) for w in df.loc[decision == "1", "candidate"].astype(str)]

    # decision == 2 自动修复判定
    auto_words: list[str] = []
    pending_rows: list[dict] = []
    sub2 = df[decision == "2"]
    for _, row in sub2.iterrows():
        word = str(row["candidate"])
        freq = int(row["doc_freq"])
        force = word in FORCE_ADD
        ok = (
            len(word) >= 2
            and bool(CJK_RE.fullmatch(word))
            and word[0] not in BOUNDARY_EXCLUDE
            and word[-1] not in BOUNDARY_EXCLUDE
            and not any(ch in INTERNAL_EXCLUDE for ch in word)
            and not START_FUNC_RE.match(word)
            and not END_FUNC_RE.search(word)
            and freq >= AUTO_MIN_DOC_FREQ
        )
        if ok or force:
            auto_words.append(word)
        else:
            pending_rows.append(
                {
                    "type": row["type"],
                    "candidate": word,
                    "doc_freq": freq,
                    "current_split": row.get("current_split", ""),
                    "reason": "force" if force else "跨词片段或含虚词，不建议入词典",
                }
            )

    # 整体重写追加区：以上次自动追加的起始词为界，避免残留坏词条
    current_lines = [l.strip() for l in DICT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    append_marker = "一起聊"
    head = current_lines[: current_lines.index(append_marker)] if append_marker in current_lines else current_lines
    dict_new = sorted((set(dict_words_raw) | set(auto_words)) - set(stop_words) - set(head))
    DICT_PATH.write_text("\n".join(head + dict_new) + ("\n" if dict_new else ""), encoding="utf-8")

    STOP_PATH.write_text("\n".join(stop_words) + ("\n" if stop_words else ""), encoding="utf-8")

    pd.DataFrame(pending_rows).to_csv(PENDING_PATH, index=False, encoding="utf-8-sig")

    lines = [
        f"decision=0 停用词: {len(stop_words)} 条 -> {STOP_PATH.name}",
        f"decision=1 领域词: {len(dict_words_raw)} 条（新增 {len([w for w in dict_words_raw if w in dict_new])}）",
        f"decision=2 自动入库: {len(auto_words)} 条；待复核: {len(pending_rows)} 条 -> {PENDING_PATH.name}",
        f"game_terms.txt 当前词条总数: {len(load_lines(DICT_PATH))}",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
