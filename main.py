# -*- coding: utf-8 -*-
"""
命令行入口：训练 + 评估 + 报告生成。

示例：
    python main.py --test-size 0.2 --n-estimators 400 --report-language zh
"""
import argparse
from pathlib import Path
import json

import config
from src import data_loader
from src import model as model_module
from src import shap_utils
from src.report_generator import generate_reports


def parse_args():
    """解析命令行参数，便于快速调参。"""
    parser = argparse.ArgumentParser(description="AI 信贷评分训练与评估")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE, help="测试集比例")
    parser.add_argument("--n-estimators", type=int, default=config.MODEL_PARAMS["n_estimators"], help="树数量")
    parser.add_argument("--max-depth", type=int, default=config.MODEL_PARAMS["max_depth"], help="树深度")
    parser.add_argument("--learning-rate", type=float, default=config.MODEL_PARAMS["learning_rate"], help="学习率")
    parser.add_argument("--report-language", type=str, default=config.DEFAULT_REPORT_LANGUAGE, choices=["zh", "en", "de"], help="报告语言")
    return parser.parse_args()


def main():
    args = parse_args()
    params = {
        **config.MODEL_PARAMS,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }

    print("📥 正在加载数据...")
    X, y = data_loader.load_data()
    # 采用布尔平均值计算正例比例，避免分类 dtype 的均值报错
    positive_rate = float((y == 1).mean())
    print(f"数据规模：{X.shape}, 目标正例比例：{positive_rate:.3f}")

    print("🤖 开始训练模型...")
    model, metrics = model_module.train_evaluate(
        X, y, params=params, test_size=args.test_size
    )
    model_path = model_module.save_model(model)
    print(f"模型已保存到：{model_path}")
    print("模型指标：", json.dumps(metrics, ensure_ascii=False, indent=2))

    print("🔍 计算 SHAP 全局重要度...")
    shap_result = shap_utils.compute_global_shap(model, X)
    importance_df = shap_result["importance"]

    print("📝 生成报告...")
    json_path, pdf_path = generate_reports(metrics, importance_df, language=args.report_language)
    print(f"JSON 报告：{json_path}")
    print(f"PDF  报告：{pdf_path}")


if __name__ == "__main__":
    main()

