# -*- coding: utf-8 -*-
"""
数据加载与缓存模块。

职责：
- 首次运行时从 OpenML 拉取 German Credit Dataset（ID=31，别名 credit-g）
- 将数据缓存为 CSV，避免重复下载
- 提供统一的 DataFrame 读取接口
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.datasets import fetch_openml

import config


def fetch_german_credit(cache_path: Path = config.CACHE_FILE) -> pd.DataFrame:
    """
    拉取并缓存 German Credit 数据集。

    Args:
        cache_path: 缓存文件路径
    Returns:
        含目标列 `class` 的 DataFrame
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)

    # 从 OpenML 下载；若网络受限，可提示用户手动放置 CSV
    dataset = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = dataset.frame
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    读取数据并分离特征与标签。

    Returns:
        X: 特征 DataFrame
        y: 标签 Series（二分类 good/bad）
    """
    df = fetch_german_credit()
    target_col = config.TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不存在，请检查数据源")

    X = df.drop(columns=[target_col])
    # 显式转为数值型，避免 Pandas 分类类型在 .mean() 等聚合时报错
    y = df[target_col].apply(lambda v: 1 if str(v).lower() == "good" else 0).astype("int64")
    return X, y

