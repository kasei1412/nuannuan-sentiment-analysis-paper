# -*- coding: utf-8 -*-
"""训练集弱标注抽检：旧弱标注 vs 否定感知弱标注，产出待复核清单。"""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path

from build_dataset import weak_label

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "comments.csv"
TAB = ROOT / "results" / "tables"
OVERRIDES = ROOT / "data" / "dict" / "label_overrides.csv"

POS_WORDS = ["好看","喜欢","爱了","绝了","惊喜","精致","解压","开心","友好","自由","精彩","满意","期待","可爱","漂亮","优秀","棒","赞","出片","养老","好玩","不错","太美","爱看","美好","美哭","好康","真好","太香","幸福","治愈","舒服","感谢","感动","永远爱","冲了","爱死","绝美","神仙","好评","真香","爱慕"]
NEG_WORDS = ["卡顿","优化差","太差","太坑","BUG","bug","崩溃","重复","无聊","难用","后悔","失望","投诉","太贵","爆率","劝退","垃圾","差劲","无语","恶心","闪退","掉帧","不满","寄了","坐牢","肝爆","服了","离谱","吐了","差评","难玩","玩不下去","退游","删了","坏了","有问题","卡成","进不去","连不上","修不好","假的"]
PRE = ["不", "没", "没有", "无"]

def negated_occ(text, word):
    """统计 word 前被否定前缀覆盖的次数；'没有' 在 '有没有X' 中不算否定。"""
    cnt = 0
    for p in PRE:
        pat = p + word
        start = 0
        while True:
            i = text.find(pat, start)
            if i < 0:
                break
            if p == "没有" and i > 0 and text[i-1] == "有":
                pass  # “有没有好看的”等疑问结构，不算否定
            else:
                cnt += 1
            start = i + 1
    # 特殊：NEG 词“有问题”的否定常写作“没有问题/没问题”
    if word == "有问题":
        for pat in ("没有问题", "没问题"):
            start = 0
            while True:
                i = text.find(pat, start)
                if i < 0:
                    break
                if pat == "没有问题" and i > 0 and text[i-1] == "有":
                    pass
                else:
                    cnt += 1
                start = i + 1
    return cnt

def polarity(text, word):
    total = text.count(word)
    neg = negated_occ(text, word)
    neg = min(neg, total)
    return total - neg, neg

def old_label(text):
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)
    if "优化" in text and any(x in text for x in ("差","不行","跟不上","垃圾","差评","求")):
        neg += 1
    if pos == 0 and neg == 0:
        return "neutral", pos, neg
    return ("positive" if pos >= neg else "negative"), pos, neg

def new_label(text):
    pos = neg = 0
    pos_hits, neg_hits = [], []
    for w in POS_WORDS:
        pc, nc = polarity(text, w)
        pos += pc; neg += nc
        pos_hits += [w] * pc
        neg_hits += ["不" + w] * nc
    for w in NEG_WORDS:
        pc, nc = polarity(text, w)
        neg += pc; pos += nc
        neg_hits += [w] * pc
        pos_hits += ["不" + w] * nc
    if "优化" in text and any(x in text for x in ("差","不行","跟不上","垃圾","差评","求")):
        neg += 1; neg_hits.append("优化+差")
    if pos == 0 and neg == 0:
        return "neutral", pos, neg, pos_hits, neg_hits
    return ("positive" if pos >= neg else "negative"), pos, neg, pos_hits, neg_hits

rows = list(csv.DictReader(RAW.open(encoding="utf-8-sig")))
diff = []
for r in rows:
    o, po, no = old_label(r["comment"])
    n, pn, nn, ph, nh = new_label(r["comment"])
    if o != n:
        diff.append({**r, "old_label": o, "new_label": n,
                     "old_pos": po, "old_neg": no, "new_pos": pn, "new_neg": nn,
                     "pos_hits": ";".join(dict.fromkeys(ph)), "neg_hits": ";".join(dict.fromkeys(nh))})

fields = ["game_name","time","comment","old_label","new_label","old_pos","old_neg","new_pos","new_neg","pos_hits","neg_hits","source_file"]
with (TAB / "label_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for d in sorted(diff, key=lambda x: (x["old_label"], x["new_label"], x["game_name"])):
        w.writerow(d)

lines = [f"训练集 {len(rows)} 条；疑似标签不一致 {len(diff)} 条", ""]
for key in [("positive","negative"), ("negative","positive")]:
    sub = [d for d in diff if (d["old_label"], d["new_label"]) == key]
    lines.append(f"===== {key[0]} -> {key[1]}（{len(sub)} 条）=====")
    for d in sub:
        lines.append(f"[{d['game_name']} | {d['time']} | {d['source_file']}]")
        lines.append(f"  文本: {d['comment'].strip()}")
        lines.append(f"  命中正: {d['pos_hits'] or '-'}  命中负: {d['neg_hits'] or '-'}  旧: {d['old_label']} -> 新: {d['new_label']}")
        lines.append("")
(TAB / "label_review_notes.txt").write_text("\n".join(lines), encoding="utf-8")

override_rows = list(csv.DictReader(OVERRIDES.open(encoding="utf-8-sig"))) if OVERRIDES.exists() else []
before_correct = sum(weak_label(r["comment"]) == r["label"] for r in override_rows)
summary = [
    {"metric": "人工复核样本数", "value": len(override_rows)},
    {"metric": "复核前弱标注一致数", "value": before_correct},
    {
        "metric": "复核前弱标注一致率",
        "value": round(before_correct / len(override_rows), 4) if override_rows else 0.0,
    },
    {"metric": "复核后一致率", "value": 1.0 if override_rows else 0.0},
]
with (TAB / "label_quality_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "value"])
    w.writeheader()
    w.writerows(summary)
print("不一致:", len(diff), dict(Counter((d["old_label"], d["new_label"]) for d in diff)))
print("已写出 label_review.csv / label_review_notes.txt / label_quality_summary.csv")
