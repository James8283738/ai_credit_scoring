# -*- coding: utf-8 -*-
"""
SHAP 解释封装。

职责：
- 提供全局特征重要度（mean | abs | summary 数据）
- 提供单样本解释（force plot 数据）
- 控制采样规模以兼顾性能
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
import shap

import config


def compute_global_shap(model, X: pd.DataFrame, sample_size: int = config.SHAP_SAMPLE_SIZE) -> Dict[str, Any]:
    """
    计算全局 SHAP 重要度。

    Args:
        model: 已训练好的 Pipeline（含 preprocess + model）
        X: 原始特征 DataFrame
        sample_size: 采样行数，避免过慢
    Returns:
        dict，包含 feature_importance DataFrame 和 summary 数组
    """
    # 采样以降低计算开销
    if len(X) > sample_size:
        X_sample = shap.utils.sample(X, sample_size, random_state=config.RANDOM_SEED)
    else:
        X_sample = X

    # 提取底层模型（xgboost）用于解释
    preprocess = model.named_steps["preprocess"]
    booster = model.named_steps["model"]
    X_transformed = preprocess.transform(X_sample)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_transformed)

    # 获取 onehot 后的特征名，便于映射
    feature_names = preprocess.get_feature_names_out()
    # 对于二分类模型，shap_values 是一个包含两个数组的列表，取第二个数组（正类）
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "importance": importance_df,
        "shap_values": shap_values,
        "X_transformed": X_transformed,
        "feature_names": feature_names
    }


def compute_single_shap(model, X_row: pd.DataFrame) -> Dict[str, Any]:
    """
    计算单样本 SHAP 解释（用于 force plot 数据）。

    Args:
        model: Pipeline
        X_row: 单行 DataFrame
    Returns:
        dict: base_value, shap_values, expected_value
    """
    preprocess = model.named_steps["preprocess"]
    booster = model.named_steps["model"]

    X_trans = preprocess.transform(X_row)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_trans)

    # 对于二分类模型，shap_values 是一个包含两个数组的列表，取第二个数组（正类）
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return {
        "base_value": float(explainer.expected_value),
        "shap_values": shap_values[0].tolist(),
        "feature_values": X_trans.toarray().tolist()[0] if hasattr(X_trans, "toarray") else X_trans.tolist()[0],
        "feature_names": preprocess.get_feature_names_out().tolist(),
    }

