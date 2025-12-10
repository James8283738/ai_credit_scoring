# -*- coding: utf-8 -*-
"""
模型训练与评估模块。

- 构建：预处理流水线 + xGBoost 分类器
- 评估：AUC、F1、KS、准确率
- 自动调优：使用 Optuna 进行超参数优化
- 持久化：保存/加载模型与预处理器
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
import xgboost as xgb
import optuna

import config
from src.feature_engineering import split_features, build_preprocess_pipeline


def _compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算 KS 值，用于信用评分常用指标。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def build_model(params: Dict) -> xgb.XGBClassifier:
    """构建 xgboost 分类器，方便统一入口。"""
    return xgb.XGBClassifier(**params)


def objective(trial, X_train, y_train, preprocess):
    """
    Optuna 目标函数，用于超参数优化。
    
    Args:
        trial: Optuna 试验对象
        X_train: 训练特征
        y_train: 训练标签
        preprocess: 预处理流水线
    Returns:
        auc: 交叉验证的 AUC 分数
    """
    # 定义超参数搜索空间
    param_space = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    
    # 构建模型
    clf = build_model(param_space)
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])
    
    # 交叉验证评估
    scores = cross_val_score(
        model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1
    )
    
    return scores.mean()


def optimize_hyperparameters(
    X, y, preprocess, n_trials: int = 50, random_state: int = config.RANDOM_SEED
) -> Dict:
    """
    使用 Optuna 优化模型超参数。
    
    Args:
        X: 特征数据
        y: 标签数据
        preprocess: 预处理流水线
        n_trials: 试验次数
        random_state: 随机种子
    Returns:
        best_params: 最佳超参数组合
    """
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # 分割训练集用于调优
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, preprocess),
        n_trials=n_trials,
        n_jobs=-1
    )
    
    return study.best_params


def train_evaluate(
    X,
    y,
    params: Dict = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
    optimize_hyperparams: bool = False,
    n_trials: int = 50,
    feature_engineering_params: Optional[Dict] = None
) -> Tuple[Pipeline, Dict]:
    """
    训练并返回模型与指标。

    Args:
        X: 特征数据
        y: 标签数据
        params: 模型参数，如果为 None 则使用默认参数
        test_size: 测试集比例
        random_state: 随机种子
        optimize_hyperparams: 是否自动优化超参数
        n_trials: 超参数优化的试验次数
        feature_engineering_params: 特征工程参数
    Returns:
        pipeline: 预处理 + 模型流水线
        metrics: dict，包含 auc/f1/ks/acc 与阈值
    """
    # 处理特征工程参数
    feature_engineering_params = feature_engineering_params or {}
    
    # 构建特征工程流水线
    num_cols, cat_cols = split_features(X)
    preprocess = build_preprocess_pipeline(num_cols, cat_cols, **feature_engineering_params)
    
    # 超参数优化
    if optimize_hyperparams:
        print("开始超参数优化...")
        params = optimize_hyperparameters(X, y, preprocess, n_trials, random_state)
        print(f"最佳超参数: {params}")
    else:
        params = params or config.MODEL_PARAMS
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 构建并训练模型
    clf = build_model(params)
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])
    model.fit(X_train, y_train)

    # 评估模型
    y_prob = model.predict_proba(X_test)[:, 1]
    default_th = config.DEFAULT_THRESHOLD
    y_pred = (y_prob >= default_th).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "ks": _compute_ks(y_test, y_prob),
        "threshold": default_th,
        "test_size": test_size,
        "n_samples": len(y),
        "hyperparameters": params,
        "optimized": optimize_hyperparams
    }
    return model, metrics


def save_model(model: Pipeline, path: Path = config.MODEL_DIR / "credit_model.pkl") -> Path:
    """保存模型流水线到指定路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = config.MODEL_DIR / "credit_model.pkl") -> Pipeline:
    """加载已训练的模型流水线。"""
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    return joblib.load(path)

