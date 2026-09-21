"""将 WebScraper 原始 CSV/XLSX 合并为统一训练表 comments.csv。"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRAPED = ROOT / "data" / "raw" / "scraped"
OUT = ROOT / "data" / "raw" / "comments.csv"
OUT_ALL = ROOT / "data" / "raw" / "comments_all.csv"
AUDIT = ROOT / "results" / "tables" / "cleaning_audit.csv"
EXCLUDED_PATH = ROOT / "data" / "dict" / "excluded_comments.csv"

GAME_CODE = {
    "qjn": ("jqnn", "奇迹暖暖"),
    "qjnn": ("jqnn", "奇迹暖暖"),
    "jqn": ("jqnn", "奇迹暖暖"),
    "jqnn": ("jqnn", "奇迹暖暖"),
    "ynn": ("yynn", "闪耀暖暖"),
    "syn": ("yynn", "闪耀暖暖"),
    "synn": ("yynn", "闪耀暖暖"),
    "sxn": ("yynn", "闪耀暖暖"),
    "yynn": ("yynn", "闪耀暖暖"),
    "wxn": ("wxnn", "无限暖暖"),
    "wxnn": ("wxnn", "无限暖暖"),
}

POS_WORDS = {
    "好看", "喜欢", "爱了", "绝了", "惊喜", "精致", "解压", "开心", "友好",
    "自由", "精彩", "满意", "期待", "可爱", "漂亮", "优秀", "棒", "赞",
    "出片", "养老", "好玩", "不错", "太美", "爱看", "美好", "美哭",
    "好康", "真好", "太香", "幸福", "治愈", "舒服", "感谢", "感动",
    "永远爱", "冲了", "爱死", "绝美", "神仙", "好评", "真香", "爱慕",
}
NEG_WORDS = {
    "卡顿", "优化差", "太差", "太坑", "BUG", "bug", "崩溃", "重复", "无聊",
    "难用", "后悔", "失望", "投诉", "太贵", "爆率", "劝退", "垃圾", "差劲",
    "无语", "恶心", "闪退", "掉帧", "不满", "寄了", "坐牢", "肝爆",
    "服了", "离谱", "吐了", "差评", "难玩", "玩不下去", "退游", "删了",
    "坏了", "有问题", "卡成", "进不去", "连不上", "修不好", "假的",
}
AD_KEYWORDS = (
    "抽奖", "福利", "转发抽", "关注抽", "代肝", "仅接官服", "信誉图",
    "结单", "vx→", "vx->", "加微信", "接单", "代练",
    "藏御堂", "返图",
    "卖号", "出号", "代出", "账号交易", "账号转移", "代售群", "邀请码",
    "私信我入群", "欢迎来问", "广告",
)

OFF_TOPIC_MARKERS = (
    "肖战", "檀健次", "宋亚轩", "时代少年团", "王者荣耀", "恋与制作人",
    "再见爱人", "李行亮", "麦琳",
)
GAME_EXPERIENCE_TERMS = {
    "游戏", "活动", "套装", "搭配", "地图", "收集", "抽卡", "充值", "优化", "更新",
    "版本", "光照", "滤镜", "引擎", "内存", "闪退", "卡顿", "美甲", "拍照",
    "开放世界", "玩法", "服装", "衣服", "副本", "剧情", "关卡", "家园",
}

# 微博「投票」自动帖：机器生成、无情感语义，一律视为无效评论。
# 典型文本如「我参与了@奇迹暖暖 发起的投票【…】，我投给了…。你也快来表态吧~」
VOTE_PATTERNS = (
    "我参与了@",
    "发起的投票",
    "你也快来表态吧",
    "来自 投票",
)

FIELDS = [
    "game",
    "game_name",
    "time",
    "comment",
    "label",
    "author",
    "source_file",
    "web_scraper_order",
    "event",
    "phase",
    "source_type",
]

# 专项事件打标：按规范文件名映射 (event, phase, source_type)。
# source_type: search=搜索博文；official_post_comment=官方预热帖评论。
EVENT_PHASE = {
    "synn240717.xlsx": ("engine_2024", "pre", "search"),
    "synn240724.xlsx": ("engine_2024", "post", "search"),
    "synn240731.xlsx": ("engine_2024", "post_supp", "search"),
    "synn260722.xlsx": ("rebirth_2026", "pre", "search"),
    "synn260731.xlsx": ("rebirth_2026", "post", "search"),
    "synn260722_cmt.xlsx": ("rebirth_2026", "pre", "official_post_comment"),
}


def detect_encoding(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            path.read_text(encoding=enc)
            return enc
        except Exception:
            continue
    return "gb18030"


def parse_filename(name: str) -> tuple[str, str, str] | None:
    """qjn191030.csv / qjn180805.xlsx / synn260722_cmt.xlsx -> (jqnn, 奇迹暖暖, 2019-10-30)"""
    stem = Path(name).stem.lower()
    m = re.match(r"^([a-z]+)(\d{6})([a-z0-9_]*)$", stem)
    if not m:
        return None
    code, yymmdd = m.group(1), m.group(2)
    if code not in GAME_CODE:
        return None
    game, game_name = GAME_CODE[code]
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    year = 2000 + int(yy)
    date = f"{year:04d}-{mm}-{dd}"
    return game, game_name, date


def normalize_time(text: str, fallback_date: str) -> str:
    text = (text or "").strip()
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})", text)
    if m:
        y, mo, d, h, mi = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{mi}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})", text)
    if m:
        mo, d, h, mi = m.groups()
        y = fallback_date[:4]
        return f"{y}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{mi}"
    m = re.search(r"(\d{2,4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", text)
    if m:
        y, mo, d, h, mi = m.groups()
        if len(y) == 2:
            y = 2000 + int(y)
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{mi}"
    if re.search(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return f"{fallback_date} 00:00"


def is_ad(text: str) -> bool:
    return any(k in text for k in AD_KEYWORDS)


def is_vote(text: str) -> bool:
    return any(p in text for p in VOTE_PATTERNS)


def is_off_topic(text: str) -> bool:
    """过滤明显的外部话题；带有游戏体验语义的比较文本仍予以保留。"""
    return any(k in text for k in OFF_TOPIC_MARKERS) and not any(
        k in text for k in GAME_EXPERIENCE_TERMS
    )


NEG_PREFIXES = ("不", "没", "没有", "无")


def _negated_occ(text: str, word: str) -> int:
    cnt = 0
    for p in NEG_PREFIXES:
        pat = p + word
        start = 0
        while True:
            i = text.find(pat, start)
            if i < 0:
                break
            if p == "没有" and i > 0 and text[i - 1] == "有":
                pass  # “有没有好看的”等疑问结构，不算否定
            else:
                cnt += 1
            start = i + 1
    if word == "有问题":
        for pat in ("没有问题", "没问题"):
            start = 0
            while True:
                i = text.find(pat, start)
                if i < 0:
                    break
                if pat == "没有问题" and i > 0 and text[i - 1] == "有":
                    pass
                else:
                    cnt += 1
                start = i + 1
    return cnt


def weak_label(text: str) -> str:
    """增强版弱标注：调用 sentiment_rules 引擎（程度副词+转折+双重否定）。"""
    try:
        from sentiment_rules import enhanced_weak_label
        label, _ = enhanced_weak_label(text)
        return label
    except ImportError:
        # 降级到旧版逻辑
        pos = neg = 0
        for w in POS_WORDS:
            total = text.count(w)
            ng = min(_negated_occ(text, w), total)
            pos += total - ng
            neg += ng
        for w in NEG_WORDS:
            total = text.count(w)
            ng = min(_negated_occ(text, w), total)
            neg += total - ng
            pos += ng
        if "优化" in text and any(x in text for x in ("差", "不行", "跟不上", "垃圾", "差评", "求")):
            neg += 1
        if pos == 0 and neg == 0:
            return "neutral"
        return "positive" if pos >= neg else "negative"


LABEL_OVERRIDE_PATH = ROOT / "data" / "dict" / "label_overrides.csv"


def load_label_overrides() -> dict[tuple[str, str], str]:
    if not LABEL_OVERRIDE_PATH.exists():
        return {}
    overrides: dict[tuple[str, str], str] = {}
    with LABEL_OVERRIDE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            overrides[(r["source_file"], r["comment"])] = r["label"]
    return overrides


def load_excluded_markers() -> dict[str, list[tuple[str, str]]]:
    if not EXCLUDED_PATH.exists():
        return {}
    excluded: dict[str, list[tuple[str, str]]] = {}
    with EXCLUDED_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            excluded.setdefault(r["source_file"], []).append((r["marker"], r["reason"]))
    return excluded


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    if SCRAPED.exists():
        files.extend(sorted(SCRAPED.glob("*.csv")))
        files.extend(sorted(SCRAPED.glob("*.xlsx")))
    return files


# WebScraper xlsx 导出时列名可能为 comment/time，也可能为点选器名 data/data2/...
# 不同 sitemap 的列含义不稳定（正文可能在 data/data2/data3/data6，时间列可能带填充空格甚至无年份），
# 因此用「内容识别」定位时间列与正文列，而不是固定列名。
_XLSX_TIME_RE = re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}")
_XLSX_TIME_NOYEAR_RE = re.compile(r"^\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}")
_XLSX_TIME_ISO_RE = re.compile(r"^\d{2,4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}")
_XLSX_TEXT_RE = re.compile(r"^(data\d*|comment|text)$")


def _detect_xlsx_cols(df: pd.DataFrame) -> tuple[str | None, str | None, set[str]]:
    """返回 (时间列, 正文列, 时间列集合)。
    - 时间列：匹配时间格式比例最高的列（作为时间输出）；
    - 正文列：在「所有非时间列」中，取非空率≥0.5 且平均长度最长的文本列；
    - 时间列集合：供调用方在「命中列为空时逐行回退」时排除时间列。
      注意：填充版时间列（尾随空格/零宽字符导致 strip 后仍很长）也要按时间格式排除，不能只排除最优时间列。"""
    time_ratio: dict[str, float] = {}
    for c in df.columns:
        vals = df[c].astype(str).str.strip()
        nonempty = vals[vals != ""]
        if nonempty.empty:
            continue
        time_ratio[c] = max(
            nonempty.str.match(_XLSX_TIME_RE).mean(),
            nonempty.str.match(_XLSX_TIME_NOYEAR_RE).mean(),
            nonempty.str.match(_XLSX_TIME_ISO_RE).mean(),
        )
    time_cols = {c for c, r in time_ratio.items() if r >= 0.5}
    time_col = max(time_ratio, key=time_ratio.get) if time_ratio else None
    if time_col is not None and time_ratio[time_col] < 0.5:
        time_col = None
    cands = [c for c in df.columns if _XLSX_TEXT_RE.fullmatch(c) and c not in time_cols]
    scored = []
    for c in cands:
        vals = df[c].astype(str).str.strip()
        nonempty = vals[vals != ""]
        if nonempty.empty:
            continue
        ratio = len(nonempty) / len(df)
        if ratio < 0.5:
            continue
        scored.append((ratio * nonempty.str.len().mean(), c))
    comment_col = max(scored)[1] if scored else None
    return time_col, comment_col, time_cols


def iter_xlsx_rows(path: Path):
    """把 WebScraper 导出的 xlsx 转为 {comment, time, raw, src_vote, order} 字典。"""
    df = pd.read_excel(path, dtype=str).fillna("")
    time_col, comment_col, time_cols = _detect_xlsx_cols(df)
    if comment_col is None:
        print(f"  跳过 xlsx（找不到评论文本列）: {path.name}")
        return
    if time_col is None:
        print(f"  提示: xlsx 未识别到时间列，将用文件名日期兜底: {path.name}")
    order_col = "web_scraper_order" if "web_scraper_order" in df.columns else "web-scraper-order"
    name_col = "name" if "name" in df.columns else None
    source_cols = [c for c in df.columns if c not in ("comment", "text", comment_col)]
    # 同一 sitemap 内不同行可能命中不同选择器（正文落在 data/data3/data10…）：
    # 命中列（comment_col）在某行为空时，回退取该行最长的非空候选文本列，避免漏行。
    fallback_cols = [c for c in df.columns if _XLSX_TEXT_RE.fullmatch(c) and c not in time_cols]
    for _, r in df.iterrows():
        comment = str(r[comment_col]).strip()
        if not comment and fallback_cols:
            cand = [(str(r[c]).strip(), c) for c in fallback_cols if str(r[c]).strip()]
            if cand:
                comment = max(cand, key=lambda x: len(x[0]))[0]
        # 来源标签列里出现独立的「投票」→ 投票自动帖
        src_vote = any(str(r[c]).strip() == "投票" for c in source_cols)
        yield {
            "comment": comment,
            "author": str(r[name_col]).strip() if name_col else "",
            "time": str(r[time_col]).strip() if time_col else "",
            "raw": " ".join(str(v) for v in r.values),
            "src_vote": src_vote,
            "order": str(r[order_col]).strip() if order_col in df.columns else "",
        }


def iter_source_rows(dest: Path):
    """按文件类型产出 {comment, time, raw, order} 字典。"""
    if dest.suffix.lower() == ".xlsx":
        yield from iter_xlsx_rows(dest)
        return
    enc = detect_encoding(dest)
    with dest.open("r", encoding=enc, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield {
                "comment": (r.get("comment") or "").strip(),
                "time": r.get("time") or "",
                "raw": " ".join(str(v) for v in r.values()),
                "order": r.get("web-scraper-order", ""),
            }


def _resolve_label(comment: str, source_file: str, overrides: dict) -> str:
    label = weak_label(comment)
    key = (source_file, comment)
    if key in overrides:
        label = overrides[key]
    return label


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build() -> Path:
    sources = iter_source_files()
    if not sources:
        raise FileNotFoundError(f"未找到爬取 CSV/XLSX。请放到 {SCRAPED}。")

    SCRAPED.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    source_counts: Counter[str] = Counter()
    filter_counts: Counter[str] = Counter()
    overrides = load_label_overrides()
    excluded_markers = load_excluded_markers()

    for src in sources:
        meta = parse_filename(src.name)
        if meta is None:
            print(f"跳过无法解析的文件名: {src.name}")
            continue
        game, game_name, file_date = meta
        dest = SCRAPED / src.name
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())

        for item in iter_source_rows(dest):
            comment = item["comment"]
            if not comment:
                filter_counts["build_empty"] += 1
                continue
            raw = item["raw"] or comment
            if is_ad(raw):
                filter_counts["build_ad"] += 1
                continue
            if is_vote(raw) or item.get("src_vote"):
                filter_counts["build_vote"] += 1
                continue
            if any(marker in comment for marker, _ in excluded_markers.get(src.name, [])):
                filter_counts["build_manual_exclusion"] += 1
                continue
            if is_off_topic(comment):
                filter_counts["build_off_topic"] += 1
                continue
            event, phase, source_type = EVENT_PHASE.get(src.name, ("", "", "search"))
            rows_out.append(
                {
                    "game": game,
                    "game_name": game_name,
                    "time": normalize_time(item["time"], file_date),
                    "comment": comment,
                    "label": _resolve_label(comment, src.name, overrides),
                    "author": item.get("author", ""),
                    "source_file": src.name,
                    "web_scraper_order": item["order"],
                    "event": event,
                    "phase": phase,
                    "source_type": source_type,
                }
            )
            source_counts[src.name] += 1

    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in rows_out:
        # 官方帖评论按 (游戏, 作者, 评论) 去重（同文本不同用户算不同评论）；
        # 搜索博文维持 (游戏, 评论) 去重。
        if r["source_type"] == "official_post_comment":
            key = (r["game"], r["author"], r["comment"])
        else:
            key = (r["game"], r["comment"])
        if key in seen:
            filter_counts["build_duplicate"] += 1
            continue
        seen.add(key)
        unique.append(r)

    labeled = [r for r in unique if r["label"] in {"positive", "negative"}]
    write_csv(OUT_ALL, unique)
    write_csv(OUT, labeled)

    audit_rows = [
        {"stage": "build", "reason": reason, "count": count}
        for reason, count in sorted(filter_counts.items())
    ]
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows, columns=["stage", "reason", "count"]).to_csv(
        AUDIT, index=False, encoding="utf-8-sig"
    )

    print(f"去广告+投票+去重后: {len(unique)} 条 -> {OUT_ALL}")
    print(f"二分类训练集: {len(labeled)} 条 -> {OUT}")
    print("来源文件计数:", dict(source_counts))
    print("按游戏(全量):", dict(Counter(r["game_name"] for r in unique)))
    print("按标签(全量):", dict(Counter(r["label"] for r in unique)))
    print("按标签(训练):", dict(Counter(r["label"] for r in labeled)))
    print("按事件/阶段(全量):", dict(Counter((r["event"], r["phase"]) for r in unique)))
    print("清洗审计:", dict(filter_counts), "->", AUDIT)
    print("说明: label 为关键词弱标注；neutral 仅保留在 comments_all.csv。")
    return OUT


if __name__ == "__main__":
    build()
