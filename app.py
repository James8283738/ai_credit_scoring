# -*- coding: utf-8 -*-
"""
Streamlit 仪表盘入口。

主要功能：
- 一键加载/训练 xGBoost 信贷评分模型
- 指标展示（AUC/F1/KS/Accuracy）与阈值调节
- Plotly 特征重要度/评分分布/单样本 SHAP 解释
- 三语报告生成并下载
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import data_loader
from src import model as model_module
from src import shap_utils
from src.report_generator import generate_reports


# ---------- UI 基础配置 ----------
st.set_page_config(
    page_title="AI 信贷评分模型",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_or_train_model():
    """
    优先加载本地模型；若不存在则快速训练一个。
    """
    model_path = config.MODEL_DIR / "credit_model.pkl"
    if model_path.exists():
        model = model_module.load_model(model_path)
        X, y = data_loader.load_data()
        metrics = None  # 无即时指标，需提示
    else:
        X, y = data_loader.load_data()
        model, metrics = model_module.train_evaluate(X, y)
        model_module.save_model(model, model_path)
    return model, metrics, X, y


def plot_feature_importance(importance_df: pd.DataFrame):
    """Plotly 横向条形图展示特征重要度。"""
    fig = px.bar(
        importance_df.head(20).iloc[::-1],
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={"mean_abs_shap": "平均绝对 SHAP", "feature": "特征"},
        title="Top20 特征重要度（SHAP）"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_score_distribution(probs: np.ndarray, threshold: float):
    """评分分布与阈值对比。"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=probs, nbinsx=30, name="违约概率"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", name="阈值")
    fig.update_layout(
        title="预测概率分布",
        xaxis_title="违约概率",
        yaxis_title="样本数"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_metrics(metrics: dict):
    """以卡片方式展示核心指标。"""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC", f"{metrics.get('auc', 0):.3f}")
    col2.metric("F1", f"{metrics.get('f1', 0):.3f}")
    col3.metric("KS", f"{metrics.get('ks', 0):.3f}")
    col4.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")


def main():
    st.title("🏦 AI 信贷评分模型（German Credit）")
    st.caption("特征工程 → xGBoost → SHAP → Plotly/Streamlit → 三语报告")

    with st.sidebar:
        st.header("⚙️ 训练/加载")
        re_train = st.button("重新训练模型")
        language = st.selectbox("报告语言", ["中文 (zh)", "Deutsch (de)", "English (en)"])
        threshold = st.slider("违约概率阈值", 0.1, 0.9, float(config.DEFAULT_THRESHOLD), 0.01)
        st.markdown("---")
        st.header("🧪 样本选择")
        selected_index = st.number_input("样本索引（0 开始）", min_value=0, value=0, step=1)

    # 缓存加载/训练
    if re_train or "model" not in st.session_state:
        with st.spinner("正在加载/训练模型..."):
            model, metrics, X, y = load_or_train_model()
            if metrics is None:
                # 若是加载路径，重新评估一次
                _, metrics = model_module.train_evaluate(X, y)
            st.session_state.model = model
            st.session_state.metrics = metrics
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.shap_global = shap_utils.compute_global_shap(model, X)

    model = st.session_state.model
    metrics = st.session_state.metrics
    X = st.session_state.X
    y = st.session_state.y
    shap_global = st.session_state.shap_global

    # 指标与分布
    st.subheader("模型指标")
    display_metrics(metrics)

    # 概率分布
    with st.spinner("计算预测概率..."):
        probs = model.predict_proba(X)[:, 1]
    plot_score_distribution(probs, threshold)

    # 特征重要度
    st.subheader("特征重要度（SHAP）")
    plot_feature_importance(shap_global["importance"])

    # 单样本解释
    st.subheader("单客户 SHAP 解释")
    selected_index = min(int(selected_index), len(X) - 1)
    sample_row = X.iloc[[selected_index]]
    shap_single = shap_utils.compute_single_shap(model, sample_row)

    # 显示原始数据（可选）
    with st.expander("查看原始特征值"):
        st.write(sample_row)
    
    # 显示SHAP值详细信息
    with st.expander("查看SHAP值详细数据"):
        # 最简单直接的方式：直接显示各个字段
        st.write("基准值 (Base Value):", float(shap_single["base_value"]))
        
        st.write("\n前5个特征的SHAP值:")
        for i in range(5):
            if i < len(shap_single["shap_values"]):
                st.write(f"  - {shap_single['feature_names'][i]}: {shap_single['shap_values'][i]:.4f}")
        
        # 添加一个简单的表格显示
        st.write("\n特征影响表格:")
        df = pd.DataFrame({
            "特征名称": shap_single["feature_names"][:10],
            "SHAP值": shap_single["shap_values"][:10]
        })
        st.table(df)
    
    # 计算并显示预测结果
    proba = model.predict_proba(sample_row)[0, 1]
    is_default = "高风险" if proba >= threshold else "低风险"
    st.metric("预测违约概率", f"{proba:.3f}")
    st.metric("风险等级", is_default)
    
    # 显示前10个最具影响力的特征
    st.subheader("关键特征影响")
    # 对SHAP值和特征名进行排序，取绝对值最大的前10个
    sorted_indices = np.argsort(np.abs(shap_single["shap_values"]))[::-1][:10]
    sorted_shap = [shap_single["shap_values"][i] for i in sorted_indices]
    sorted_features = [shap_single["feature_names"][i] for i in sorted_indices]
    sorted_values = [shap_single["feature_values"][i] for i in sorted_indices]
    
    # 创建DataFrame并显示
    impact_df = pd.DataFrame({
        "特征名称": sorted_features,
        "特征值": sorted_values,
        "SHAP值": sorted_shap,
        "影响方向": ["正向" if s > 0 else "负向" for s in sorted_shap]
    })
    
    # 使用Plotly创建条形图可视化
    fig = px.bar(
        impact_df,
        x="SHAP值",
        y="特征名称",
        orientation="h",
        color="SHAP值",
        color_continuous_scale="RdBu_r",
        hover_data=["特征值", "影响方向"],
        title="特征影响程度（前10个）"
    )
    fig.update_layout(
        xaxis_title="SHAP值（影响程度）",
        yaxis_title="特征名称",
        yaxis_categoryorder="total ascending"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 报告生成
    st.markdown("---")
    st.subheader("📄 生成报告")
    if st.button("生成 JSON + PDF 报告"):
        lang_code = "zh" if "中文" in language else ("de" if "Deutsch" in language else "en")
        json_path, pdf_path = generate_reports(metrics, shap_global["importance"], language=lang_code)
        st.success("报告已生成")
        with open(json_path, "r", encoding="utf-8") as f:
            st.download_button("下载 JSON 报告", f.read(), file_name=Path(json_path).name, mime="application/json")
        with open(pdf_path, "rb") as f:
            st.download_button("下载 PDF 报告", f.read(), file_name=Path(pdf_path).name, mime="application/pdf")

    with st.expander("术语与金融解释"):
        st.markdown("""
        - **AUC**：区分好/坏客户的能力，越接近 1 越好。  
        - **KS**：好坏客户累积分布最大差异，>0.2 通常可接受。  
        - **阈值**：将违约概率转为好/坏客户标签的分界，可按业务偏好调节。  
        - **SHAP**：衡量特征对单个预测的边际贡献，便于解释。  
        """)


if __name__ == "__main__":
    main()

