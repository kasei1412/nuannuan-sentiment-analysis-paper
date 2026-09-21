# 分析流程 | Analysis Pipeline

本目录包含游戏用户体验分析作品集的完整实现。

## 方法链路

1. **数据接入**：统一读取 CSV 与 WebScraper XLSX 导出文件。
2. **文本预处理**：中文清洗、话题标签处理、`jieba` 分词和停用词过滤。
3. **弱监督标注**：结合正负面词典、否定感知规则与广告/投票过滤。
4. **情感分类**：TF-IDF + Naive Bayes / Logistic Regression / Linear SVM。
5. **主题建模**：LDA、主题数选择和主题稳定性分析。
6. **体验诊断**：按游戏、时间窗口和版本事件输出聚合对比。

## 关键文件

```text
src/
├── build_dataset.py       # 数据统一读取与弱标注
├── preprocess.py          # 文本清洗与中文分词
├── sentiment_ml.py        # TF-IDF 情感分类实验
├── lda_topics.py          # LDA 主题建模
├── event_comparison.py    # 事件前后窗口比较
├── make_sample_data.py    # 脱敏演示数据
└── run_all.py             # 一键运行入口
```

## 技术关键词

`Python` · `pandas` · `scikit-learn` · `jieba` · `TF-IDF` · `SVM` · `LDA` · `Cross-validation` · `Text Mining`

## 运行

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/run_all.py
```

原始数据和逐行输出仅允许在本地使用，不应提交到 GitHub；仓库中的结果目录只保留聚合指标和图表。
