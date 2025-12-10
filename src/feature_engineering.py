# -*- coding: utf-8 -*-
"""
增强版特征工程模块。

职责：
- 区分数值/类别特征
- 数值：缺失填补 + 标准化/归一化
- 类别：缺失填补 + one-hot 编码/WOE 编码
- 特征选择：基于重要性的特征筛选
- 新特征生成：交互特征、多项式特征
- 提供可复用的预处理流水线（用于训练与推理）
"""
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, 
    PolynomialFeatures, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb


def split_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    将特征按 dtype 分成数值列与类别列。

    Args:
        df: 原始特征 DataFrame
    Returns:
        num_cols: 数值列名列表
        cat_cols: 类别列名列表
    """
    num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


class WOEEncoder:
    """
    Weight of Evidence (WOE) 编码器，适用于信用评分模型。
    注意：仅用于训练，推理时需使用预先计算的 WOE 值。
    """
    def __init__(self, smooth: float = 0.5):
        self.smooth = smooth  # 平滑参数，避免除零
        self.woe_mapping: Dict[str, Dict[str, float]] = {}  # 存储每个特征的 WOE 映射
        self.iv_values: Dict[str, float] = {}  # 存储每个特征的 IV 值

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """计算每个特征的 WOE 映射。"""
        self.woe_mapping = {}
        self.iv_values = {}
        
        # 确保 X 是 DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        for col in X.columns:
            # 计算每个类别下的好坏样本数
            cross = pd.crosstab(X[col], y, margins=False)
            cross.columns = ['good', 'bad'] if 0 in cross.columns else ['bad', 'good']
            
            # 添加平滑
            cross['good'] = cross['good'] + self.smooth
            cross['bad'] = cross['bad'] + self.smooth
            
            # 计算概率
            total_good = cross['good'].sum()
            total_bad = cross['bad'].sum()
            cross['p_good'] = cross['good'] / total_good
            cross['p_bad'] = cross['bad'] / total_bad
            
            # 计算 WOE 和 IV
            cross['woe'] = np.log(cross['p_good'] / cross['p_bad'])
            cross['iv'] = (cross['p_good'] - cross['p_bad']) * cross['woe']
            
            # 存储映射和 IV
            self.woe_mapping[col] = cross['woe'].to_dict()
            self.iv_values[col] = cross['iv'].sum()
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用预计算的 WOE 值转换特征。"""
        # 确保 X 是 DataFrame
        is_ndarray = isinstance(X, np.ndarray)
        if is_ndarray:
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        X_woe = X.copy()
        for col in X.columns:
            if col in self.woe_mapping:
                # 对于未知类别，使用该特征的平均 WOE
                mean_woe = np.mean(list(self.woe_mapping[col].values()))
                X_woe[col] = X_woe[col].map(self.woe_mapping[col]).fillna(mean_woe)
        
        # 如果输入是 ndarray，输出也保持 ndarray
        if is_ndarray:
            return X_woe.values
        return X_woe

    def get_iv_values(self) -> Dict[str, float]:
        """返回所有特征的 IV 值。"""
        return self.iv_values


def create_interaction_features(X: pd.DataFrame, interaction_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    生成特征交互项。
    
    Args:
        X: 原始特征 DataFrame
        interaction_pairs: 要交互的特征对列表
    Returns:
        X_with_interactions: 包含交互特征的新 DataFrame
    """
    X_new = X.copy()
    for col1, col2 in interaction_pairs:
        if col1 in X.columns and col2 in X.columns:
            interaction_name = f"{col1}_x_{col2}"
            X_new[interaction_name] = X[col1] * X[col2]
    return X_new


def build_preprocess_pipeline(
    num_cols: List[str], 
    cat_cols: List[str],
    use_woe: bool = False,
    use_polynomial: bool = False,
    polynomial_degree: int = 2,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    feature_selection: bool = False,
    feature_selection_threshold: float = 0.01
) -> Pipeline:
    """
    构建增强版预处理流水线。

    Args:
        num_cols: 数值列
        cat_cols: 类别列
        use_woe: 是否使用 WOE 编码（仅适用于训练）
        use_polynomial: 是否生成多项式特征
        polynomial_degree: 多项式特征的次数
        interaction_pairs: 要生成的特征交互对
        feature_selection: 是否进行特征选择
        feature_selection_threshold: 特征选择的阈值
    Returns:
        完整的预处理流水线
    """
    # 数值特征处理
    numeric_transformers = []
    
    # 基础数值处理
    numeric_base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    numeric_transformers.append(("numeric_base", numeric_base, num_cols))
    
    # 多项式特征（如果启用）
    if use_polynomial and num_cols:
        polynomial_features = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False))
        ])
        numeric_transformers.append(("polynomial", polynomial_features, num_cols))
    
    # 类别特征处理
    if use_woe:
        # WOE 编码（需要 y 标签，仅适用于训练阶段）
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("woe", WOEEncoder())
        ])
    else:
        # 默认使用 One-Hot 编码
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    
    # 构建基础预处理
    preprocess = ColumnTransformer(
        transformers=[
            *numeric_transformers,
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="passthrough"
    )
    
    # 构建完整流水线
    pipeline_steps = []
    
    # 特征交互（如果提供了交互对）
    if interaction_pairs:
        interaction_transformer = FunctionTransformer(
            create_interaction_features, 
            kw_args={"interaction_pairs": interaction_pairs}
        )
        pipeline_steps.append(("interaction", interaction_transformer))
    
    # 添加基础预处理
    pipeline_steps.append(("preprocess", preprocess))
    
    # 特征选择（如果启用）
    if feature_selection:
        selector = SelectFromModel(
            xgb.XGBClassifier(), 
            threshold=feature_selection_threshold,
            prefit=False
        )
        pipeline_steps.append(("feature_selection", selector))
    
    return Pipeline(steps=pipeline_steps)


def get_feature_importance(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    获取模型的特征重要性。
    
    Args:
        model: 训练好的模型流水线
        feature_names: 原始特征名称列表
    Returns:
        feature_importance: 包含特征名称和重要性的 DataFrame
    """
    # 检查模型是否有特征选择步骤
    if "feature_selection" in model.named_steps:
        selector = model.named_steps["feature_selection"]
        importances = selector.estimator_.feature_importances_
        # 获取选择后的特征索引
        selected_indices = selector.get_support(indices=True)
        
        # 构建选择后的特征名称映射
        preprocess = model.named_steps["preprocess"]
        transformer_names = [name for name, _, _ in preprocess.transformers_]
        
        # 重建特征名称
        processed_feature_names = []
        for name, _, cols in preprocess.transformers_:
            if name == "numeric_base" or name == "numeric_transformer":
                processed_feature_names.extend(cols)
            elif name == "polynomial":
                # 多项式特征名称比较复杂，这里简化处理
                poly_names = [f"poly_{col}" for col in cols]
                processed_feature_names.extend(poly_names)
            elif name == "cat":
                # One-Hot 编码后的特征名称
                ohe = preprocess.named_transformers_[name].named_steps["onehot"]
                cat_features = ohe.get_feature_names_out(cols)
                processed_feature_names.extend(cat_features)
            elif name == "remainder":
                processed_feature_names.extend(cols)
        
        # 获取选择的特征名称
        selected_features = [processed_feature_names[i] for i in selected_indices]
        
        return pd.DataFrame({
            "feature": selected_features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
    else:
        # 如果没有特征选择，直接从模型获取
        model_step = None
        for step_name, step in reversed(model.named_steps.items()):
            if hasattr(step, "feature_importances_"):
                model_step = step
                break
        
        if model_step is None:
            raise ValueError("模型中没有可获取特征重要性的步骤")
            
        importances = model_step.feature_importances_
        
        # 构建特征名称映射
        preprocess = model.named_steps["preprocess"]
        processed_feature_names = []
        for name, _, cols in preprocess.transformers_:
            if name == "numeric_base" or name == "numeric_transformer":
                processed_feature_names.extend(cols)
            elif name == "polynomial":
                # 多项式特征名称比较复杂，这里简化处理
                poly_names = [f"poly_{col}" for col in cols]
                processed_feature_names.extend(poly_names)
            elif name == "cat":
                # One-Hot 编码后的特征名称
                if "onehot" in preprocess.named_transformers_[name].named_steps:
                    ohe = preprocess.named_transformers_[name].named_steps["onehot"]
                    cat_features = ohe.get_feature_names_out(cols)
                    processed_feature_names.extend(cat_features)
                elif "woe" in preprocess.named_transformers_[name].named_steps:
                    # WOE 编码后的特征名称与原特征相同
                    processed_feature_names.extend(cols)
            elif name == "remainder":
                processed_feature_names.extend(cols)
        
        return pd.DataFrame({
            "feature": processed_feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

