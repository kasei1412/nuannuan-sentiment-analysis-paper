# -*- coding: utf-8 -*-
"""从清洗文本中挖掘候选领域词短语，输出待人工筛选清单。

两类候选：
1. token：当前分词结果中高频、但不在 game_terms.txt / stopwords.txt 中的词；
2. ngram：高频字符 n-gram、单独分词会被切开、且首尾不是虚词，提示可能缺少词典词条。

输出 results/tables/domain_term_candidates.csv，
人工在 decision 列填写：领域词典 / LDA停用 / 同义词→标准词 / 忽略。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "comments_clean_all.csv"
DICT = ROOT / "data" / "dict" / "game_terms.txt"
STOP = ROOT / "data" / "dict" / "stopwords.txt"
OUT = ROOT / "results" / "tables" / "domain_term_candidates.csv"

TOKEN_MIN_DOC_FREQ = 8
NGRAM_MIN_DOC_FREQ = 8
NGRAM_MAX_LEN = 5
NGRAM_MIN_SOURCES = 4
TOP_N_PER_TYPE = 200

GAME_NAMES = ("奇迹暖暖", "闪耀暖暖", "无限暖暖")
CJK_RE = re.compile(r"[\u4e00-\u9fa5]{2,}")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fa5]")

# 首尾为这些字符的 n-gram 视为跨词片段，直接排除
BOUNDARY_EXCLUDE = set(
    "的了是在是我有和就都也都还这么那你不没吧呢啊呀哦么什很被把让给从到与及或而等着过得之上下里外前后完起"
)

# 明显通用/功能性的高频词，建议作为 LDA 停用词而非领域词
AUTO_STOP_HINTS = {
    "还是", "感觉", "今天", "知道", "这次", "这个", "一下", "现在", "之后", "时候",
    "真的", "就是", "还有", "这样", "不要", "大家", "多少", "怎么", "为什么", "可以",
    "比较", "然后", "一个", "之前", "自己", "一直", "一起", "出来", "已经", "一次",
    "直接", "觉得", "发现", "终于", "开始", "有点", "一样", "时间", "希望", "问题",
    "有没有", "不能", "这么", "一点", "什么",
}


def load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def has_cjk(text: str) -> bool:
    return bool(CJK_CHAR_RE.search(text))


def touches_game_name(gram: str) -> bool:
    return any(name in gram or gram in name for name in GAME_NAMES)


def main() -> None:
    game_terms = load_lines(DICT)
    stopwords = load_lines(STOP)

    df = pd.read_csv(CLEAN)
    texts = df["comment_clean"].astype(str).tolist()
    games = df["game"].astype(str).tolist()

    # ---------- 1) 高频 token ----------
    token_doc_freq: Counter[str] = Counter()
    token_total: Counter[str] = Counter()
    token_game_freq: dict[str, Counter[str]] = {}
    for text, game in zip(texts, games):
        seen: set[str] = set()
        for word in jieba.lcut(text):
            word = word.strip()
            if len(word) < 2 or not has_cjk(word):
                continue
            if word in game_terms or word in stopwords:
                continue
            token_total[word] += 1
            seen.add(word)
        for word in seen:
            token_doc_freq[word] += 1
            token_game_freq.setdefault(word, Counter())[game] += 1

    token_rows = []
    for word, freq in token_doc_freq.most_common(TOP_N_PER_TYPE * 2):
        if freq < TOKEN_MIN_DOC_FREQ:
            break
        gf = token_game_freq.get(word, Counter())
        example = next((t for t in texts if word in t), "")
        token_rows.append(
            {
                "type": "token",
                "candidate": word,
                "doc_freq": freq,
                "total_count": token_total[word],
                "jqnn_df": gf.get("jqnn", 0),
                "yynn_df": gf.get("yynn", 0),
                "wxnn_df": gf.get("wxnn", 0),
                "current_split": "整词",
                "in_game_dict": int(word in game_terms),
                "in_stopwords": int(word in stopwords),
                "example": example[:80],
                "suggestion": "疑似通用词，可考虑LDA停用" if word in AUTO_STOP_HINTS else "",
                "decision": "",
            }
        )
    token_rows = token_rows[:TOP_N_PER_TYPE]

    # ---------- 2) 高频 n-gram ----------
    ngram_doc_freq: Counter[str] = Counter()
    ngram_total: Counter[str] = Counter()
    ngram_game_freq: dict[str, Counter[str]] = {}
    ngram_sources: dict[str, set[str]] = {}
    for text, game, source in zip(texts, games, df["source_file"].astype(str).tolist()):
        for chunk in CJK_RE.findall(text):
            seen: set[str] = set()
            for n in range(2, NGRAM_MAX_LEN + 1):
                for i in range(len(chunk) - n + 1):
                    gram = chunk[i : i + n]
                    if gram[0] in BOUNDARY_EXCLUDE or gram[-1] in BOUNDARY_EXCLUDE:
                        continue
                    if touches_game_name(gram):
                        continue
                    seen.add(gram)
                    ngram_total[gram] += 1
            for gram in seen:
                ngram_doc_freq[gram] += 1
                ngram_sources.setdefault(gram, set()).add(source)
                ngram_game_freq.setdefault(gram, Counter())[game] += 1

    existing_tokens = set(token_doc_freq)
    ngram_rows = []
    for gram, freq in ngram_doc_freq.most_common(TOP_N_PER_TYPE * 6):
        if freq < NGRAM_MIN_DOC_FREQ:
            break
        if gram in existing_tokens or gram in game_terms or gram in stopwords:
            continue
        if len(ngram_sources.get(gram, set())) < NGRAM_MIN_SOURCES:
            continue
        split_result = [w for w in jieba.lcut(gram) if w.strip()]
        if split_result == [gram]:
            continue
        if len(split_result) > 1 and all(w in stopwords for w in split_result):
            continue
        gf = ngram_game_freq.get(gram, Counter())
        example = next((t for t in texts if gram in t), "")
        ngram_rows.append(
            {
                "type": "ngram",
                "candidate": gram,
                "doc_freq": freq,
                "total_count": ngram_total[gram],
                "jqnn_df": gf.get("jqnn", 0),
                "yynn_df": gf.get("yynn", 0),
                "wxnn_df": gf.get("wxnn", 0),
                "current_split": "|".join(split_result),
                "in_game_dict": 0,
                "in_stopwords": 0,
                "example": example[:80],
                "suggestion": "疑似领域短语，可加入词典",
                "decision": "",
            }
        )
        if len(ngram_rows) >= TOP_N_PER_TYPE:
            break

    rows = sorted(token_rows + ngram_rows, key=lambda r: (-r["doc_freq"], r["type"], r["candidate"]))
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"token 候选: {len(token_rows)} 条；ngram 候选: {len(ngram_rows)} 条")
    print(f"已写出: {OUT}")


if __name__ == "__main__":
    main()
