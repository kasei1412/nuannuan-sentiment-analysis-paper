"""生成演示用评论数据（结构与真实采集表一致，便于先跑通流程）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "comments.csv"

SAMPLES = [
    # 奇迹暖暖 - 积极
    ("jqnn", "2018-08-05 20:11", "奇迹暖暖新套装染色方案太好看了，星之海终于复刻", "positive"),
    ("jqnn", "2018-08-05 21:02", "今天体力换到了心仪礼包，搭配自由度很高很养老", "positive"),
    ("jqnn", "2019-10-31 12:30", "收集齐了套装，染色系统真的很解压", "positive"),
    ("jqnn", "2019-10-31 18:45", "竞技场JJC打得挺开心，兑换也很友好", "positive"),
    # 奇迹暖暖 - 消极
    ("jqnn", "2018-08-05 22:10", "套装获取太难了，礼包太贵不友好", "negative"),
    ("jqnn", "2019-10-31 19:20", "体力不够用，抽奖爆率太差了", "negative"),
    # 闪耀暖暖 - 积极
    ("yynn", "2019-10-31 10:05", "闪耀暖暖美甲设计太绝了，3D建模很精致", "positive"),
    ("yynn", "2019-10-31 15:40", "万圣节活动打卡很有趣，设计感很强", "positive"),
    ("yynn", "2021-06-15 09:18", "部件拆分后搭配更自由，开门效果好看", "positive"),
    ("yynn", "2021-06-15 20:33", "耀暖建模细节不错，拍照也很出片", "positive"),
    # 闪耀暖暖 - 消极
    ("yynn", "2019-10-31 21:50", "部件拆分机制太坑，爆率太低了", "negative"),
    ("yynn", "2021-06-15 22:01", "移动端卡顿严重，优化太差玩不下去", "negative"),
    # 无限暖暖 - 积极
    ("wxnn", "2025-02-01 14:22", "无限暖暖开放世界地图解谜很有意思，拍照系统爱了", "positive"),
    ("wxnn", "2025-02-01 19:08", "苏暖家园布置很好看，随机事件挺惊喜", "positive"),
    ("wxnn", "2025-05-31 11:15", "花焰季活动氛围不错，留影功能好用", "positive"),
    ("wxnn", "2025-05-31 16:40", "探索自由度高，希望以后能联机", "positive"),
    # 无限暖暖 - 消极
    ("wxnn", "2025-02-01 21:59", "BUG太多了烟花都炸了，开放世界任务太重复", "negative"),
    ("wxnn", "2025-05-31 20:10", "抽卡机制争议大，内容不够深度优化差", "negative"),
    ("wxnn", "2025-05-31 21:30", "地图指引不清晰，任务重复性太高", "negative"),
    ("wxnn", "2025-02-01 22:40", "性能优化不行，卡顿和BUG影响体验", "negative"),
]


def main() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SAMPLES, columns=["game", "time", "comment", "label"])
    # 扩充到更接近课程演示规模（重复加噪声时间戳）
    rows = []
    for i in range(8):
        tmp = df.copy()
        tmp["time"] = pd.to_datetime(tmp["time"]) + pd.to_timedelta(i, unit="h")
        tmp["comment"] = tmp["comment"] + ("" if i == 0 else f" 补充体验{i}")
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RAW, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows -> {RAW}")


if __name__ == "__main__":
    main()
