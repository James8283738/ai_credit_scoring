# -*- coding: utf-8 -*-
"""
模型模块的测试用例。
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import model as model_module
from src import data_loader


def test_build_model():
    """测试模型构建功能。"""
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42
    }
    
    clf = model_module.build_model(params)
    
    # 检查模型是否正确构建
    assert hasattr(clf, 'fit')
    assert hasattr(clf, 'predict')
    assert hasattr(clf, 'predict_proba')


def test_train_evaluate_basic():
    """测试基本的模型训练与评估功能。"""
    # 加载测试数据
    X, y = data_loader.load_data()
    
    # 使用少量数据加速测试
    X_small = X.head(100)
    y_small = y.head(100)
    
    model, metrics = model_module.train_evaluate(X_small, y_small, test_size=0.2)
    
    # 检查返回结果
    assert isinstance(model, Pipeline)
    assert isinstance(metrics, dict)
    
    # 检查关键指标是否存在且有效
    assert 'auc' in metrics
    assert 'f1' in metrics
    assert 'accuracy' in metrics
    assert 'ks' in metrics
    
    assert metrics['auc'] > 0
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['accuracy'] <= 1
    assert metrics['ks'] >= 0


def test_train_evaluate_with_params():
    """测试使用自定义参数的模型训练。"""
    # 加载测试数据
    X, y = data_loader.load_data()
    X_small = X.head(100)
    y_small = y.head(100)
    
    custom_params = {
        "n_estimators": 50,
        "learning_rate": 0.2,
        "max_depth": 2,
        "random_state": 42
    }
    
    model, metrics = model_module.train_evaluate(X_small, y_small, params=custom_params, test_size=0.2)
    
    assert isinstance(model, Pipeline)
    assert 'auc' in metrics


def test_train_evaluate_with_feature_engineering_params():
    """测试带有特征工程参数的模型训练。"""
    # 加载测试数据
    X, y = data_loader.load_data()
    X_small = X.head(100)
    y_small = y.head(100)
    
    # 测试使用高级特征工程参数
    feature_engineering_params = {
        'use_polynomial': True,
        'polynomial_degree': 2,
        'feature_selection': False
    }
    
    model, metrics = model_module.train_evaluate(
        X_small, 
        y_small, 
        test_size=0.2,
        feature_engineering_params=feature_engineering_params
    )
    
    assert isinstance(model, Pipeline)
    assert 'auc' in metrics


def test_model_prediction():
    """测试模型预测功能。"""
    # 加载测试数据
    X, y = data_loader.load_data()
    X_small = X.head(50)
    y_small = y.head(50)
    
    model, _ = model_module.train_evaluate(X_small, y_small, test_size=0.2)
    
    # 测试预测概率
    y_prob = model.predict_proba(X_small[:10])[:, 1]
    assert len(y_prob) == 10
    assert all(0 <= prob <= 1 for prob in y_prob)
    
    # 测试预测类别
    y_pred = model.predict(X_small[:10])
    assert len(y_pred) == 10
    assert set(y_pred).issubset({0, 1})  # 确保预测结果是0或1


def test_optimize_hyperparameters_smoke():
    """测试超参数优化功能的冒烟测试。"""
    # 加载测试数据
    X, y = data_loader.load_data()
    X_small = X.head(100)
    y_small = y.head(100)
    
    # 创建简单的预处理流水线
    num_cols, cat_cols = model_module.split_features(X_small)
    preprocess = model_module.build_preprocess_pipeline(num_cols, cat_cols)
    
    # 只运行少量试验以加速测试
    best_params = model_module.optimize_hyperparameters(X_small, y_small, preprocess, n_trials=2)
    
    assert isinstance(best_params, dict)
    assert len(best_params) > 0
    
    # 检查关键超参数是否存在
    assert 'n_estimators' in best_params
    assert 'learning_rate' in best_params
    assert 'max_depth' in best_params


def test_train_evaluate_with_hyperparameter_optimization():
    """测试启用超参数优化的模型训练（只运行少量试验以加速测试）。"""
    # 加载测试数据
    X, y = data_loader.load_data()
    X_small = X.head(100)
    y_small = y.head(100)
    
    # 使用少量试验以加速测试
    model, metrics = model_module.train_evaluate(
        X_small, 
        y_small, 
        test_size=0.2,
        optimize_hyperparams=True,
        n_trials=2
    )
    
    assert isinstance(model, Pipeline)
    assert isinstance(metrics, dict)
    assert 'auc' in metrics
    assert metrics['optimized'] is True  # 检查是否标记为已优化
