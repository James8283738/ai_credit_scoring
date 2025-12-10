# -*- coding: utf-8 -*-
"""
特征工程模块的测试用例。
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    split_features,
    build_preprocess_pipeline,
    WOEEncoder,
    create_interaction_features,
    get_feature_importance
)


def test_split_features():
    """测试特征分割功能。"""
    # 创建测试数据
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'cat1': ['A', 'B', 'A', 'B', 'C'],
        'cat2': [True, False, True, False, True]
    })
    
    num_cols, cat_cols = split_features(df)
    
    assert set(num_cols) == {'num1', 'num2'}
    assert set(cat_cols) == {'cat1', 'cat2'}


def test_build_preprocess_pipeline_basic():
    """测试基本预处理流水线构建。"""
    # 创建测试数据
    df = pd.DataFrame({
        'num1': [1, 2, np.nan, 4, 5],
        'cat1': ['A', 'B', 'A', np.nan, 'C'],
        'cat2': [True, False, True, False, True]
    })
    
    num_cols = ['num1']
    cat_cols = ['cat1', 'cat2']
    
    pipeline = build_preprocess_pipeline(num_cols, cat_cols)
    
    # 检查流水线是否正确构建
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) > 0
    
    # 测试流水线是否可以正常转换数据
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 5
    assert not np.isnan(transformed).any()  # 确保没有缺失值


def test_build_preprocess_pipeline_with_woe():
    """测试带有WOE编码的预处理流水线。"""
    # 创建测试数据
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'cat1': ['A', 'B', 'A', 'B', 'C']
    })
    y = pd.Series([0, 1, 0, 1, 0])  # 标签数据
    
    num_cols = ['num1']
    cat_cols = ['cat1']
    
    pipeline = build_preprocess_pipeline(num_cols, cat_cols, use_woe=True)
    
    assert isinstance(pipeline, Pipeline)
    
    # 测试流水线转换
    transformed = pipeline.fit_transform(df, y)
    assert transformed.shape[0] == 5


def test_build_preprocess_pipeline_with_feature_selection():
    """测试带有特征选择的预处理流水线。"""
    # 创建测试数据
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [6, 7, 8, 9, 10],
        'num3': [11, 12, 13, 14, 15],
        'cat1': ['A', 'B', 'A', 'B', 'C']
    })
    y = pd.Series([0, 1, 0, 1, 0])  # 标签数据
    
    num_cols = ['num1', 'num2', 'num3']
    cat_cols = ['cat1']
    
    pipeline = build_preprocess_pipeline(
        num_cols, cat_cols, 
        use_woe=False,
        feature_selection=True
    )
    
    assert isinstance(pipeline, Pipeline)
    
    # 测试流水线转换
    transformed = pipeline.fit_transform(df, y)
    assert transformed.shape[0] == 5


def test_create_interaction_features():
    """测试交互特征生成功能。"""
    # 创建测试数据
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [6, 7, 8, 9, 10],
        'cat1': ['A', 'B', 'A', 'B', 'C']
    })
    
    # 转换类别特征为数值
    df['cat1'] = df['cat1'].map({'A': 0, 'B': 1, 'C': 2})
    
    # 生成交互特征
    interaction_df = create_interaction_features(df, interaction_pairs=[('num1', 'num2'), ('num1', 'cat1')])
    
    # 检查交互特征是否生成
    assert 'num1_x_num2' in interaction_df.columns
    assert 'num1_x_cat1' in interaction_df.columns
    
    # 验证交互特征计算是否正确
    assert interaction_df['num1_x_num2'].tolist() == [6, 14, 24, 36, 50]
    assert interaction_df['num1_x_cat1'].tolist() == [0, 2, 0, 4, 10]


def test_woe_encoder():
    """测试WOE编码器功能。"""
    # 创建测试数据
    X = pd.DataFrame({
        'cat1': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C'],
        'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])  # 标签数据
    
    # 初始化并拟合WOE编码器
    encoder = WOEEncoder()
    encoder.fit(X, y)
    
    # 转换数据
    X_encoded = encoder.transform(X)
    
    assert X_encoded.shape == X.shape
    assert not np.isnan(X_encoded.values).any()  # 确保没有缺失值
    
    # 测试IV值
    iv_values = encoder.get_iv_values()
    assert isinstance(iv_values, dict)
    assert 'cat1' in iv_values
    assert 'cat2' in iv_values
    assert iv_values['cat1'] > 0
    assert iv_values['cat2'] > 0
