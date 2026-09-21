"""文本清洗与中文分词。"""
from __future__ import annotations

import re
from pathlib import Path

import jieba
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "comments.csv"
PROCESSED = ROOT / "data" / "processed" / "comments_clean.csv"
DICT = ROOT / "data" / "dict" / "game_terms.txt"
STOP = ROOT / "data" / "dict" / "stopwords.txt"
AUDIT = ROOT / "results" / "tables" / "cleaning_audit.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
AT_RE = re.compile(r"@\S+")
TOPIC_RE = re.compile(r"#([^#]+)#")
GAME_TOPIC_RE = re.compile(r"(?:奇迹暖暖|闪耀暖暖|无限暖暖)(?:搭配|美服|交易|高分攻略组|互粉)?超话")
NON_TEXT_RE = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9，。！？、；：,.!?;:\s]")
PLATFORM_RE = re.compile(r"(?:展开|收起)[A-Za-z]?|微博视频|微博正文")
AD_KEYWORDS = (
    "抽奖",
    "福利",
    "转发抽",
    "关注抽",
    "代肝",
    "仅接官服",
    "信誉图",
    "结单",
    "vx→",
    "vx->",
    "加微信",
    "接单",
    "代练",
    "卖号",
    "出号",
    "代出",
    "账号交易",
    "账号转移",
    "代售群",
    "邀请码",
    "私信我入群",
    "欢迎来问",
    "广告",
)

OFF_TOPIC_MARKERS = (
    "肖战",
    "檀健次",
    "宋亚轩",
    "时代少年团",
    "王者荣耀",
    "恋与制作人",
)
GAME_CONTEXT_TERMS = {
    "游戏", "游戏体验", "更新", "版本", "活动", "套装", "搭配", "地图", "收集",
    "抽卡", "充值", "优化", "光照", "滤镜", "引擎", "内存", "闪退", "卡顿",
    "美甲", "拍照", "开放世界", "奇迹暖暖", "闪耀暖暖", "无限暖暖",
}
LOW_INFORMATION_RE = re.compile(
    r"^(?:打卡|签到|报到|报道|冒个泡|来了|滴|滴滴|转发|点赞)[!！。.,，~～ ]*$"
)
TOKEN_NORMALIZATION = {
    "BUG": "bug",
    "Bug": "bug",
    "JJC": "jjc",
    "3D": "3d",
    "PC": "pc",
}


def load_resources() -> set[str]:
    if DICT.exists():
        jieba.load_userdict(str(DICT))
    stopwords: set[str] = set()
    if STOP.exists():
        stopwords = {
            line.strip()
            for line in STOP.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return stopwords


def clean_text(text: str) -> str:
    text = str(text)

    def keep_topic_content(match: re.Match[str]) -> str:
        content = match.group(1)
        content = re.sub(r"奇迹暖暖|闪耀暖暖|无限暖暖|超话", " ", content)
        return content

    # 移除官方宣传 tag 噪声（须在 TOPIC_RE 提取 tag 内容之前，覆盖空格/标点变体）
    text = re.sub(r"无限暖暖\s*全球公测", " ", text)
    text = re.sub(r"收集美好的开放世界[!！?？。，,\s]*", " ", text)
    text = re.sub(r"(?:和美好的世界牵手|与美好牵手)", " ", text)
    text = re.sub(r"全球公测", " ", text)
    text = re.sub(r"向全世界安利", " ", text)

    text = TOPIC_RE.sub(keep_topic_content, text)
    text = GAME_TOPIC_RE.sub(" ", text)
    text = text.replace("超话", " ")
    text = URL_RE.sub(" ", text)
    text = AT_RE.sub(" ", text)
    text = text.replace("网页链接", " ").replace("展开全文", " ")
    text = PLATFORM_RE.sub(" ", text)
    text = NON_TEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_ad(text: str) -> bool:
    return any(k in text for k in AD_KEYWORDS)


def is_off_topic(text: str) -> bool:
    """仅过滤明显的非目标话题，保留可能存在游戏比较语义的文本。"""
    return any(k in text for k in OFF_TOPIC_MARKERS) and not any(
        k in text for k in GAME_CONTEXT_TERMS
    )


def is_low_information(text: str) -> bool:
    if LOW_INFORMATION_RE.fullmatch(text):
        return True
    if "我关注了" in text and "加入" in text and "超话" in text:
        return True
    if "等级达到" in text and "加入" in text and "超话" in text:
        return True
    return False


def tokenize(text: str, stopwords: set[str]) -> str:
    tokens = [w.strip() for w in jieba.lcut(text) if w.strip()]
    tokens = [w for w in tokens if w not in stopwords and len(w) > 1]
    tokens = [TOKEN_NORMALIZATION.get(w, w) for w in tokens]
    return " ".join(tokens)


def preprocess(
    input_path: Path = RAW,
    output_path: Path = PROCESSED,
    require_label: bool = True,
) -> "pd.DataFrame":
    if not input_path.exists():
        raise FileNotFoundError(
            f"未找到原始数据: {input_path}\n请先放入 comments.csv 或运行 build_dataset.py / make_sample_data.py"
        )

    stopwords = load_resources()
    audit_counts: dict[str, int] = {}

    def audit(reason: str, count: int = 1) -> None:
        audit_counts[reason] = audit_counts.get(reason, 0) + count

    df = pd.read_csv(input_path)
    required = {"game", "time", "comment"}
    if require_label:
        required.add("label")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    before = len(df)
    df = df.dropna(subset=["comment"]).copy()
    audit("preprocess_empty", before - len(df))
    if require_label and "label" in df.columns:
        before = len(df)
        df = df[df["label"].isin(["positive", "negative"])]
        audit("preprocess_neutral", before - len(df))
    df["comment_clean"] = df["comment"].map(clean_text)
    before = len(df)
    df = df[~df["comment_clean"].map(is_ad)]
    audit("preprocess_ad", before - len(df))
    before = len(df)
    df = df[~df["comment_clean"].map(is_off_topic)]
    audit("preprocess_off_topic", before - len(df))
    df["tokens"] = df["comment_clean"].map(lambda x: tokenize(x, stopwords))
    before = len(df)
    df = df[df["comment_clean"].str.len() >= 2]
    audit("preprocess_short", before - len(df))
    before = len(df)
    df = df[~df["comment_clean"].map(is_low_information)]
    audit("preprocess_low_information", before - len(df))
    before = len(df)
    df = df[df["tokens"].str.contains(r"[\u4e00-\u9fa5A-Za-z0-9]", regex=True, na=False)]
    audit("preprocess_empty_tokens", before - len(df))
    before = len(df)
    df = df.drop_duplicates(subset=["comment_clean"])
    audit("preprocess_duplicate", before - len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit_path = AUDIT
    audit_stage = f"preprocess:{input_path.name}"
    existing = pd.DataFrame(columns=["stage", "reason", "count"])
    if audit_path.exists():
        existing = pd.read_csv(audit_path)
    new_audit = pd.DataFrame(
        [{"stage": "preprocess", "reason": k, "count": v} for k, v in sorted(audit_counts.items())]
    )
    new_audit["stage"] = audit_stage
    stale_stages = {"preprocess", audit_stage}
    pd.concat([existing[~existing["stage"].isin(stale_stages)], new_audit], ignore_index=True).to_csv(
        audit_path, index=False, encoding="utf-8-sig"
    )
    print(f"预处理完成: {len(df)} 条 -> {output_path}")
    return df


if __name__ == "__main__":
    preprocess()
