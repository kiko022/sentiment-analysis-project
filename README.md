# IMDB Sentiment Analysis

一个用于电影评论情感分类的深度学习项目，使用双向LSTM模型。

## 项目结构

```
sentiment-analysis-project/
├── data/                # 数据目录
├── models/              # 保存的模型
├── notebooks/           # Jupyter notebooks
├── logs/                # 训练日志和错误分析
├── data_preprocessing.py  # 数据预处理
├── train.py             # 模型训练
├── error_analysis.py    # 错误分析
└── requirements.txt     # 依赖包
```

## 训练命令

```bash
python train.py --lr 0.001 --batch_size 32 --epochs 20 --seed 42
```

## 环境

- Python 3.9+
- TensorFlow 2.13
- CUDA 11.8 (可选，用于GPU加速)
