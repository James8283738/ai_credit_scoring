# -*- coding: utf-8 -*-
"""
全局配置文件，集中管理路径、模型超参、随机种子等。
在 Streamlit / 命令行模式下均会引用，便于统一调整。
"""
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

# 创建必要目录
for _path in (DATA_DIR, REPORT_DIR, MODEL_DIR):
    _path.mkdir(parents=True, exist_ok=True)

# 数据配置
DATA_SOURCE = "openml://credit-g"  # openml 数据集 id: 31
CACHE_FILE = DATA_DIR / "german_credit.csv"
TARGET_COL = "class"

# 训练/验证配置
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # 在训练集中划分验证

# xGBoost 默认超参数，可通过命令行/Streamlit 覆盖
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_SEED,
}

# 阈值与报告配置
DEFAULT_THRESHOLD = 0.5
DEFAULT_REPORT_LANGUAGE = "zh"  # zh / de / en

# SHAP 配置
SHAP_SAMPLE_SIZE = 800  # 控制 SHAP 计算的样本量，避免过慢

