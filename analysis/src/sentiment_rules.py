# -*- coding: utf-8 -*-
"""增强版弱标注引擎：程度副词加权、转折切分、双重否定识别。

替换 build_dataset.py 中原有的 weak_label() 函数。
仅影响弱标注阶段，不影响 TF-IDF + SVM 情感分类模型。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 情感词表（正面 / 负面）
# ---------------------------------------------------------------------------

POS_WORDS = {
    # 原有
    "好看", "喜欢", "爱了", "绝了", "惊喜", "精致", "解压", "开心", "友好",
    "自由", "精彩", "满意", "期待", "可爱", "漂亮", "优秀", "棒", "赞",
    "出片", "养老", "好玩", "不错", "太美", "爱看", "美好", "美哭",
    "好康", "真好", "太香", "幸福", "治愈", "舒服", "感谢", "感动",
    "永远爱", "冲了", "爱死", "绝美", "神仙", "好评", "真香", "爱慕",
    # 扩充：外观与审美
    "好看", "美丽", "漂亮", "绝美", "惊艳", "精致", "精美", "优雅",
    "华丽", "梦幻", "仙气", "少女心", "高颜值", "耐看", "养眼",
    # 扩充：玩法与体验
    "好玩", "有趣", "上头", "沉浸", "解谜", "探索", "自由度", "开放世界",
    "良心", "诚意", "用心", "创新", "突破", "进步", "优化好", "流畅",
    # 扩充：社交与情感表达
    "温馨", "暖心", "感动", "治愈", "陪伴", "回忆", "青春", "情怀",
    "安利", "种草", "真香", "上分", "白嫖", "白嫖党", "佛系",
    # 扩充：评价用语
    "好评", "五星", "满分", "推荐", "值得", "入股", "不亏", "值回",
    "神作", "天花板", "顶级", "一流", "顶尖", "完美", "无敌",
    # 扩充：网络流行语
    "yyds", "绝绝子", "拿捏", "杀疯了", "赢麻了", "笑死", "可爱到爆炸",
}

NEG_WORDS = {
    # 原有
    "卡顿", "优化差", "太差", "太坑", "BUG", "bug", "崩溃", "重复", "无聊",
    "难用", "后悔", "失望", "投诉", "太贵", "爆率", "劝退", "垃圾", "差劲",
    "无语", "恶心", "闪退", "掉帧", "不满", "寄了", "坐牢", "肝爆",
    "服了", "离谱", "吐了", "差评", "难玩", "玩不下去", "退游", "删了",
    "坏了", "有问题", "卡成", "进不去", "连不上", "修不好", "假的",
    # 扩充：技术问题
    "卡死", "黑屏", "白屏", "闪退", "掉线", "断连", "加载慢", "发热",
    "烫手", "耗电", "占内存", "存储不足", "兼容", "适配差", "模糊",
    "马赛克", "像素", "锯齿", "穿模", "贴图", "建模丑",
    # 扩充：氪金与付费
    "逼氪", "骗氪", "氪金", "吃相", "割韭菜", "套路", "坑钱", "无底洞",
    "保底", "歪了", "非酋", "天价", "溢价", "捆绑销售",
    # 扩充：运营与策划
    "策划", "运营差", "客服", "敷衍", "不作为", "冷处理", "装死",
    "画饼", "虚假宣传", "货不对板", "偷工减料", "半成品", "测试服",
    # 扩充：玩家情绪
    "恶心", "想吐", "反胃", "难受", "心累", "疲惫", "厌倦", "腻了",
    "弃坑", "卸载", "退坑", "脱坑", "劝退", "避雷", "踩雷", "雷点",
    "下头", "无语", "离谱", "抽象", "绷不住", "蚌埠住", "麻了",
    # 扩充：对比负面
    "倒退", "不如以前", "越来越差", "越做越烂", "一代不如一代",
}

# ---------------------------------------------------------------------------
# 程度副词权重表
# ---------------------------------------------------------------------------
DEGREE_ADVERBS = {
    "太": 2.0, "超": 1.8, "非常": 2.0, "特别": 1.8, "极其": 2.5,
    "最": 2.2, "超级": 2.0, "十分": 1.8, "相当": 1.6, "很": 1.5,
    "挺": 1.3, "蛮": 1.3, "比较": 1.2, "有点": 0.6, "有些": 0.6,
    "稍微": 0.4, "略微": 0.4, "一点点": 0.3, "真的": 1.4,
    "实在太": 2.3, "简直": 1.8, "真是": 1.3,
}

# ---------------------------------------------------------------------------
# 转折连词
# ---------------------------------------------------------------------------
TRANSITION_WORDS = ("但是", "但", "不过", "然而", "可是", "却", "只是", "就是")

# 否定前缀
NEG_PREFIXES = ("不", "没", "没有", "无", "别", "未")

# 双重否定模式（整体取反）
DOUBLE_NEG_PATTERNS = [
    re.compile(r"不是不"),
    re.compile(r"并没有不"),
    re.compile(r"没有不"),
    re.compile(r"不是没"),
    re.compile(r"并非不"),
]

# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def _has_double_negation(text: str, word_pos: int) -> bool:
    """检查 word 前方是否存在双重否定结构。"""
    window = text[max(0, word_pos - 8):word_pos]
    return any(p.search(window) for p in DOUBLE_NEG_PATTERNS)


def _is_negated(text: str, word_pos: int) -> bool:
    """检查 word 前方是否存在否定前缀（排除双重否定）。"""
    if _has_double_negation(text, word_pos):
        return False  # 双重否定=肯定，不算被否定
    window = text[max(0, word_pos - 4):word_pos]
    return any(window.endswith(p) or p == window for p in NEG_PREFIXES)


def _get_degree_multiplier(text: str, word_pos: int) -> float:
    """检查 word 前方是否存在程度副词，返回权重乘数（默认 1.0）。"""
    best = 1.0
    window = text[max(0, word_pos - 6):word_pos]
    for adv, weight in DEGREE_ADVERBS.items():
        if window.endswith(adv):
            best = max(best, weight)
    return best


def _is_after_transition(text: str, word_pos: int) -> bool:
    """检查 word 是否位于转折连词之后（同句内）。"""
    before = text[:word_pos]
    for tw in TRANSITION_WORDS:
        pos = before.rfind(tw)
        if pos >= 0:
            # 检查转折词后到当前词之间是否有句号（如有则不在同一子句）
            segment = before[pos + len(tw):]
            if "。" not in segment and "！" not in segment and "？" not in segment:
                return True
    return False


def _count_word_occurrences(text: str, word: str) -> list[int]:
    """返回 word 在 text 中所有出现位置的起始索引列表。"""
    positions = []
    start = 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def enhanced_weak_label(text: str) -> tuple[str, float]:
    """增强版弱标注，返回 (label, confidence_score)。"""
    pos_score = 0.0
    neg_score = 0.0

    for polarity_set, score_ref in [(POS_WORDS, "pos"), (NEG_WORDS, "neg")]:
        for word in polarity_set:
            positions = _count_word_occurrences(text, word)
            for wpos in positions:
                multiplier = 1.0
                # 1) 程度副词
                multiplier *= _get_degree_multiplier(text, wpos)
                # 2) 否定/双重否定
                negated = _is_negated(text, wpos)
                double_neg = _has_double_negation(text, wpos)
                if double_neg:
                    # 双重否定 → 极性取反、保持权重
                    pass  # 稍后在下方处理翻转
                elif negated:
                    # 单重否定 → 极性翻转
                    if score_ref == "pos":
                        neg_score += multiplier * 0.8
                    else:
                        pos_score += multiplier * 0.8
                    continue
                # 3) 转折加强
                if _is_after_transition(text, wpos):
                    multiplier *= 2.0
                # 双重否定 → 极性翻转
                if double_neg:
                    if score_ref == "pos":
                        neg_score += multiplier
                    else:
                        pos_score += multiplier
                else:
                    if score_ref == "pos":
                        pos_score += multiplier
                    else:
                        neg_score += multiplier

    # 额外规则：优化+负面词组合
    if "优化" in text and any(x in text for x in ("差", "不行", "跟不上", "垃圾", "差评", "求")):
        neg_score += 1.5

    total = pos_score + neg_score
    if total < 0.3:
        return "neutral", 0.0
    confidence = abs(pos_score - neg_score) / total
    label = "positive" if pos_score >= neg_score else "negative"
    return label, round(confidence, 4)
