"""按游戏代际进行 LDA 主题建模。"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "comments_clean.csv"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "results" / "models"
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

GAME_NAME = {"jqnn": "奇迹暖暖", "yynn": "闪耀暖暖", "wxnn": "无限暖暖"}
GAME_SELF_REFS = {
    "jqnn": {"奇迹暖暖", "奇迹", "暖暖", "奇暖"},
    "yynn": {"闪耀暖暖", "闪耀", "暖暖", "闪暖"},
    "wxnn": {"无限暖暖", "无限", "暖暖", "无暖", "暖五"},
}
EXTRA_STOP_PATH = ROOT / "data" / "dict" / "lda_stopwords_extra.txt"
LDA_GENERIC_STOPWORDS = {
    "更新", "还是", "这次", "这个", "一下", "现在", "之后", "时候", "真的",
    "就是", "还有", "这样", "不要", "知道", "需要", "大家", "多少", "怎么",
    "为什么", "可以", "比较", "然后", "一个", "热巴", "迪丽", "包出", "王一博",
}
if EXTRA_STOP_PATH.exists():
    LDA_GENERIC_STOPWORDS |= {
        line.strip()
        for line in EXTRA_STOP_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

# LDA 层同义词归一：同一概念的不同表达统一为一个词（仅作用于主题分析，不影响情感模型）
LDA_SYNONYMS = {"狗叠": "叠纸", "服装": "套装", "衣服": "套装", "裙子": "套装"}


def pick_font() -> str | None:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def fit_lda(
    docs: list[list[str]],
    num_topics: int = 3,
    passes: int = 15,
    random_state: int = 42,
) -> tuple[LdaModel, corpora.Dictionary]:
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=1, no_above=0.85)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        alpha="auto",
        eta="auto",
    )
    return model, dictionary


def select_topic_count(
    docs: list[list[str]],
    max_topics: int = 5,
    passes: int = 15,
    coherence_max_docs: int = 200,
    coherence_method: str = "c_v",
) -> tuple[LdaModel, corpora.Dictionary, int, list[dict[str, float | int]]]:
    """用主题一致性选择 K，并返回候选 K 的可复核记录。"""
    max_k = min(max_topics, max(2, len(docs) // 50))
    candidates = range(2, max_k + 1)
    scores: list[dict[str, float | int]] = []
    fitted: dict[int, tuple[LdaModel, corpora.Dictionary]] = {}
    step = max(1, len(docs) // coherence_max_docs)
    coherence_texts = docs[::step][:coherence_max_docs]
    for k in candidates:
        model, dictionary = fit_lda(docs, num_topics=k, passes=passes)
        fitted[k] = (model, dictionary)
        try:
            if coherence_method == "u_mass":
                coherence_model = CoherenceModel(
                    model=model,
                    corpus=[dictionary.doc2bow(doc) for doc in coherence_texts],
                    dictionary=dictionary,
                    coherence=coherence_method,
                )
            else:
                coherence_model = CoherenceModel(
                    model=model,
                    texts=coherence_texts,
                    dictionary=dictionary,
                    coherence=coherence_method,
                    topn=10,
                )
            coherence_score = float(coherence_model.get_coherence())
        except Exception as exc:  # noqa: BLE001
            print(f"主题一致性计算失败 K={k}: {exc}")
            coherence_score = float("nan")
        scores.append({"k": k, "coherence_cv": coherence_score})

    valid = [r for r in scores if pd.notna(r["coherence_cv"])]
    selected = int(max(valid, key=lambda r: r["coherence_cv"])["k"]) if valid else 3
    if selected not in fitted:
        selected = min(fitted)
    return (*fitted[selected], selected, scores)


def topic_similarity(model_a: LdaModel, model_b: LdaModel, topn: int = 10) -> float:
    """按主题词集合的最大 Jaccard 匹配估计不同种子下的稳定性。"""
    sets_a = [set(w for w, _ in model_a.show_topic(tid, topn=topn)) for tid in range(model_a.num_topics)]
    sets_b = [set(w for w, _ in model_b.show_topic(tid, topn=topn)) for tid in range(model_b.num_topics)]
    if not sets_a or not sets_b:
        return 0.0
    scores = []
    for a in sets_a:
        scores.append(
            max((len(a & b) / len(a | b) if a | b else 0.0) for b in sets_b)
        )
    return sum(scores) / len(scores)


def make_topic_docs(
    token_series,
    extra_stopwords: set[str] | None = None,
    drop_substrings: set[str] | None = None,
    synonym_map: dict[str, str] | None = None,
) -> list[list[str]]:
    """为 LDA 构造文档；额外停用词、子串剔除与同义词归一仅作用于主题分析，不影响情感模型。"""
    excluded = extra_stopwords or set()
    substrs = drop_substrings or set()
    synonyms = synonym_map or {}
    return [
        [
            synonyms.get(word, word)
            for word in str(tokens).split()
            if word and word not in excluded and not any(s in word for s in substrs)
        ]
        for tokens in token_series
        if str(tokens).strip()
    ]


def distinctive_terms(
    model: LdaModel, dictionary: corpora.Dictionary, topic_id: int, topn: int
) -> list[tuple[str, float]]:
    """用词项概率与主题专属性联合排序，降低跨主题高频词的干扰。"""
    phi = model.get_topics()
    topic_mass = phi.sum(axis=0) + 1e-12
    candidates = []
    for word_id, probability in enumerate(phi[topic_id]):
        exclusivity = float(probability / topic_mass[word_id])
        score = float(probability * exclusivity)
        candidates.append((score, dictionary[word_id], float(probability)))
    candidates.sort(reverse=True)
    return [(word, probability) for _, word, probability in candidates[:topn]]


def topic_table(
    model: LdaModel, dictionary: corpora.Dictionary, topn: int = 10
) -> pd.DataFrame:
    rows = []
    for tid in range(model.num_topics):
        terms = distinctive_terms(model, dictionary, tid, topn)
        raw_terms = model.show_topic(tid, topn=topn)
        rows.append(
            {
                "topic_id": tid,
                "keywords": " ".join([w for w, _ in terms]),
                "weights": ";".join([f"{w}:{weight:.4f}" for w, weight in terms]),
                "raw_keywords": " ".join([w for w, _ in raw_terms]),
            }
        )
    return pd.DataFrame(rows)


def save_wordcloud(model: LdaModel, game_key: str, topic_id: int) -> None:
    font = pick_font()
    freqs = dict(model.show_topic(topic_id, topn=30))
    wc = WordCloud(
        width=800,
        height=500,
        background_color="white",
        font_path=font,
    ).generate_from_frequencies(freqs)
    out = FIG_DIR / f"wordcloud_{game_key}_topic{topic_id}.png"
    wc.to_file(str(out))


def run_lda(data_path: Path = PROCESSED, num_topics: int = 3) -> None:
    df = pd.read_csv(data_path)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_topics = []
    selection_rows = []
    stability_rows = []
    for game, sub in df.groupby("game"):
        topic_stopwords = {GAME_NAME.get(game, game)} | LDA_GENERIC_STOPWORDS
        name = GAME_NAME.get(game, game)
        docs = make_topic_docs(
            sub["tokens"].tolist(),
            topic_stopwords,
            drop_substrings=GAME_SELF_REFS.get(game, {name}),
            synonym_map=LDA_SYNONYMS,
        )
        if len(docs) < 3:
            print(f"跳过 {game}: 文档过少")
            continue
        max_topics = min(num_topics + 2, 5)
        model, dictionary, k, scores = select_topic_count(
            docs, max_topics=max_topics, passes=15, coherence_method="u_mass"
        )
        for score in scores:
            selection_rows.append(
                {
                    "game": game,
                    "game_name": GAME_NAME.get(game, game),
                    "k": score["k"],
                    "coherence_method": "u_mass",
                    "excluded_terms": " ".join(sorted(topic_stopwords)),
                    "coherence_cv": score["coherence_cv"],
                    "selected": int(score["k"] == k),
                }
            )
        seed_models = [model]
        for seed in (43, 44):
            seed_models.append(
                fit_lda(docs, num_topics=k, passes=15, random_state=seed)[0]
            )
        pair_scores = [
            topic_similarity(seed_models[i], seed_models[j])
            for i in range(len(seed_models))
            for j in range(i + 1, len(seed_models))
        ]
        stability_rows.append(
            {
                "game": game,
                "game_name": GAME_NAME.get(game, game),
                "selected_k": k,
                "seeds": "42,43,44",
                "mean_top10_jaccard": round(sum(pair_scores) / len(pair_scores), 4),
            }
        )
        table = topic_table(model, dictionary)
        table.insert(0, "game", game)
        table.insert(1, "game_name", GAME_NAME.get(game, game))
        all_topics.append(table)

        # 保存选中的 LDA 模型与词典，供主题-情感联合分析复用（保证与已发布主题表一致）
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / f"lda_{game}.pkl", "wb") as fh:
            pickle.dump(
                {
                    "model": model,
                    "dictionary": dictionary,
                    "topic_stopwords": topic_stopwords,
                    "drop_substrings": GAME_SELF_REFS.get(game, {name}),
                    "synonym_map": LDA_SYNONYMS,
                },
                fh,
            )

        for tid in range(model.num_topics):
            try:
                save_wordcloud(model, str(game), tid)
            except Exception as exc:  # noqa: BLE001
                print(f"词云生成失败 {game}-{tid}: {exc}")

        print(f"\n[{GAME_NAME.get(game, game)}] LDA 主题：")
        print(table[["topic_id", "keywords"]].to_string(index=False))

    if all_topics:
        out = pd.concat(all_topics, ignore_index=True)
        out.to_csv(TABLE_DIR / "lda_topics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(selection_rows).to_csv(
            TABLE_DIR / "lda_model_selection.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(stability_rows).to_csv(
            TABLE_DIR / "lda_topic_stability.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n已保存: {TABLE_DIR / 'lda_topics.csv'}")


if __name__ == "__main__":
    run_lda()
