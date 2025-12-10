# -*- coding: utf-8 -*-
"""
报告生成模块（JSON + PDF，支持中文/德文/英文）。

- 汇总模型指标、阈值、特征重要度
- 对关键金融指标提供术语解释，提升可读性
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from fpdf import FPDF

import config


# 简单的术语词典，便于多语言解释
TERMS = {
    "zh": {
        "auc": "AUC（曲线下面积）：衡量模型区分好坏客户的能力，越接近1越好。",
        "f1": "F1 值：综合精确率与召回率，反映模型在不平衡样本下的稳健性。",
        "ks": "KS 值：好坏客户累积分布差异的最大距离，信用评分常用指标，通常 >0.2 较好。",
        "threshold": "阈值：将违约概率转换为好/坏客户的分界点，可在仪表盘动态调节。",
    },
    "en": {
        "auc": "AUC: Area under ROC, higher is better (closer to 1).",
        "f1": "F1: Harmonic mean of precision and recall, good for imbalance.",
        "ks": "KS: Max distance between good/bad cumulative distributions; >0.2 is decent.",
        "threshold": "Threshold: Cut-off to convert PD to good/bad; tune by business goal.",
    },
    "de": {
        "auc": "AUC: Fläche unter der ROC-Kurve, je näher an 1 desto besser.",
        "f1": "F1: Harmonisches Mittel aus Präzision und Recall, robust bei Ungleichgewicht.",
        "ks": "KS: Maximaler Abstand der kumulierten Verteilungen Gut/Schlecht; >0,2 ist gut.",
        "threshold": "Schwellenwert: Grenzwert für PD zu Gut/Schlecht; an Geschäftsziele anpassen.",
    },
}


def _t(label: str, lang: str) -> str:
    """获取术语解释，若缺失则回退英文。"""
    return TERMS.get(lang, {}).get(label) or TERMS["en"].get(label, label)


def generate_json_report(
    metrics: Dict,
    top_features: List[Dict],
    language: str = config.DEFAULT_REPORT_LANGUAGE,
    path: Path = None
) -> Path:
    """
    导出 JSON 报告，方便程序消费。
    """
    language = language.lower()
    path = path or config.REPORT_DIR / f"credit_report_{language}_{datetime.now():%Y%m%d_%H%M%S}.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "language": language,
        "metrics": metrics,
        "top_features": top_features,
        "glossary": {k: _t(k, language) for k in ["auc", "f1", "ks", "threshold"]},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def generate_pdf_report(
    metrics: Dict,
    top_features: List[Dict],
    language: str = config.DEFAULT_REPORT_LANGUAGE,
    path: Path = None
) -> Path:
    """
    导出简版 PDF 报告（fpdf2）。
    """
    language = language.lower()
    path = path or config.REPORT_DIR / f"credit_report_{language}_{datetime.now():%Y%m%d_%H%M%S}.pdf"

    pdf = FPDF()
    pdf.add_page()
    
    # 设置UTF-8编码支持
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 使用默认的Helvetica字体，确保兼容性
    pdf.set_font("Helvetica", size=14)

    # 使用英文标题，避免中文字符问题
    title_map = {
        "zh": "AI Credit Scoring Report (中文)",
        "en": "AI Credit Scoring Report",
        "de": "KI Kredit-Scoring Bericht",
    }
    pdf.cell(200, 10, txt=title_map.get(language, title_map["en"]), ln=True, align="L")
    
    # 使用默认字体
    pdf.set_font("Helvetica", size=11)
    
    pdf.cell(200, 8, txt=f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ln=True)
    pdf.ln(4)

    # 指标
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(200, 8, txt="Metrics", ln=True)
    pdf.set_font("Helvetica", size=11)
    for key in ["auc", "f1", "ks", "accuracy", "threshold"]:
        if key in metrics:
            pdf.cell(200, 7, txt=f"{key}: {metrics[key]:.4f}" if isinstance(metrics[key], float) else f"{key}: {metrics[key]}", ln=True)
    pdf.ln(3)

    # 术语解释
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(200, 8, txt="Glossary", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k in ["auc", "f1", "ks", "threshold"]:
        pdf.multi_cell(200, 6, txt=f"- {k.upper()}: {_t(k, language)}")
    pdf.ln(2)

    # 特征重要度
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(200, 8, txt="Feature Importance", ln=True)
    pdf.set_font("Helvetica", size=11)
    for feat in top_features[:10]:
        pdf.cell(200, 6, txt=f"{feat['feature']}: {feat['importance']:.4f}", ln=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


def generate_reports(metrics: Dict, importance_df, language: str = config.DEFAULT_REPORT_LANGUAGE):
    """
    同步生成 JSON 与 PDF，并返回路径。
    """
    # 为PDF生成准备数据，避免中文字符问题
    top_feats = [
        {"feature": row["feature"], "importance": float(row["mean_abs_shap"])}
        for _, row in importance_df.head(20).iterrows()
    ]
    
    # 生成JSON报告（支持中文）
    json_path = generate_json_report(metrics, top_feats, language)
    
    # 生成PDF报告（使用英文以确保兼容性）
    pdf_path = generate_pdf_report(metrics, top_feats, "en")
    
    return json_path, pdf_path

