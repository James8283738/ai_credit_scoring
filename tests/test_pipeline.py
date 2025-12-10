# -*- coding: utf-8 -*-
"""
端到端流水线测试：验证完整的数据加载、特征工程与模型训练流程。
"""
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from src import data_loader
from src import model as model_module
from src import feature_engineering as fe


def test_data_loading():
    """测试数据加载功能。"""
    X, y = data_loader.load_data()
    
    # 检查数据类型
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    # 检查数据形状
    assert X.shape[0] > 0
    assert X.shape[0] == y.shape[0]
    
    # 检查标签分布
    assert set(y.unique()) == {0, 1}  # 二分类问题


def test_train_pipeline_smoke():
    """冒烟测试：完整训练流程能返回模型和指标。"""
    X, y = data_loader.load_data()
    model, metrics = model_module.train_evaluate(X, y, test_size=0.3)
    
    # 检查返回结果
    assert isinstance(model, Pipeline)
    assert isinstance(metrics, dict)
    
    # 检查关键指标
    assert metrics["auc"] > 0
    assert metrics["ks"] >= 0
    
    # 预测功能测试
    probs = model.predict_proba(X[:5])[:, 1]
    assert probs.shape[0] == 5
    assert all(0 <= prob <= 1 for prob in probs)


def test_train_pipeline_with_feature_engineering():
    """测试带有高级特征工程的完整训练流程。"""
    X, y = data_loader.load_data()
    
    # 使用部分数据加速测试
    X_small = X.head(200)
    y_small = y.head(200)
    
    # 配置高级特征工程
    feature_engineering_params = {
        'use_polynomial': True,
        'polynomial_degree': 2,
        # 只选择前两个数值特征生成交互项
        'interaction_pairs': [('age', 'income')] if 'age' in X.columns and 'income' in X.columns else None
    }
    
    model, metrics = model_module.train_evaluate(
        X_small, 
        y_small, 
        test_size=0.3,
        feature_engineering_params=feature_engineering_params
    )
    
    assert isinstance(model, Pipeline)
    assert 'auc' in metrics
    assert metrics['auc'] > 0


def test_pipeline_integration():
    """测试流水线各个组件的集成功能。"""
    # 1. 加载数据
    X, y = data_loader.load_data()
    X_small = X.head(100)
    y_small = y.head(100)
    
    # 2. 特征工程
    num_cols, cat_cols = fe.split_features(X_small)
    
    # 3. 构建预处理流水线
    preprocess = fe.build_preprocess_pipeline(
        num_cols, 
        cat_cols,
        use_woe=False
    )
    
    # 4. 构建模型
    clf = model_module.build_model({
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42
    })
    
    # 5. 组合成完整流水线
    full_pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])
    
    # 6. 训练与预测
    full_pipeline.fit(X_small, y_small)
    y_prob = full_pipeline.predict_proba(X_small[:10])[:, 1]
    y_pred = full_pipeline.predict(X_small[:10])
    
    assert y_prob.shape[0] == 10
    assert y_pred.shape[0] == 10
    assert set(y_pred).issubset({0, 1})

