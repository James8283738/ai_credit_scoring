# AI 信贷评分模型（German Credit + xGBoost + SHAP）

## 项目概述
面向私人银行风控的端到端信贷评分示例，覆盖数据获取、特征工程、xGBoost 训练、SHAP 可解释性以及 Plotly/Streamlit 可视化仪表盘，并提供中文/德文/英文三语报告输出与 Git/GitHub 备份脚本。

## 功能特性
- 📦 数据：自动从 OpenML 拉取 German Credit Dataset（`credit-g`），本地缓存到 `data/german_credit.csv`
- 🧮 特征工程：数值标准化、分箱、类别编码（包括WOE编码）、缺失填补、特征选择、多项式特征和交互特征
- 🤖 模型：xGBoost 二分类，支持超参配置、训练/验证/测试切分、指标输出（AUC、F1、KS、准确率）
- 🔍 解释性：SHAP TreeExplainer，支持全局（summary、特征重要度）与单样本 force plot JSON
- 📊 仪表盘：Streamlit + Plotly 仪表盘（评分分布、特征重要度、单客户解释、阈值调节）
- 📝 报告：三语（中文/Deutsch/English）PDF/JSON 报告，解释金融术语，便于客户阅读
- 🧪 测试：完整的单元测试套件，包括特征工程、模型训练和流水线集成测试
- 📈 CI/CD：GitHub Actions 工作流，自动进行代码风格检查、依赖安装和测试覆盖率报告生成
- 🌐 Git 支持：`setup_git.sh` 一键初始化并推送到 GitHub

## 项目结构
```
ai_credit_scoring/
├── README.md
├── requirements.txt        # 生产依赖
├── requirements-dev.txt    # 开发依赖（测试、覆盖率等）
├── pytest.ini              # 测试配置
├── config.py
├── main.py                 # 命令行训练&评估入口
├── app.py                  # Streamlit 仪表盘
├── data/                   # 缓存数据与样例
├── reports/                # 训练与评分报告输出（JSON/PDF）
├── models/                 # 模型与特征处理器缓存
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # 数据下载与缓存
│   ├── feature_engineering.py # 特征工程与预处理
│   ├── model.py            # 训练、评估、持久化
│   ├── shap_utils.py       # SHAP 解释封装
│   └── report_generator.py # 三语报告生成
├── tests/
│   ├── test_feature_engineering.py # 特征工程测试
│   ├── test_model.py       # 模型训练测试
│   └── test_pipeline.py    # 流水线集成测试
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions 工作流
├── setup_git.sh            # Git/GitHub 自动化脚本
└── QUICK_START.md          # 快速上手与常用命令
```

## 快速开始
```bash
cd "/Users/zhonghuaxiaochu/claude agent/ai_credit_scoring"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 训练+评估+保存模型与报告
python main.py

# 启动交互式仪表盘
streamlit run app.py
```

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest

# 生成测试覆盖率报告
python -m pytest --cov=src
```

## 开发路线（满足用户需求自动更新）
- [x] 项目脚手架与 README
- [x] 数据获取与缓存（German Credit）
- [x] 特征工程 + xGBoost 训练
- [x] SHAP 解释与 Plotly 可视化
- [x] Streamlit 交互仪表盘
- [x] 三语 PDF/JSON 报告
- [x] 更多特征工程与模型调优（WOE编码、特征选择、多项式特征、交互特征）
- [x] CI/CD 与覆盖率增强（GitHub Actions 配置、测试覆盖率提升）

## 多语言报告
- 语言：`zh` / `de` / `en`
- 内容：模型指标、阈值设定、关键特征解释、金融术语注释、风险建议
- 路径：生成的 JSON/PDF 会保存在 `reports/`

## GitHub 备份
- 初学者可运行 `./setup_git.sh` 一键初始化、添加远程并推送
- 详细命令可参考 `QUICK_START.md`

## 测试
### 运行测试
```bash
# 运行所有测试
python -m pytest

# 运行指定测试文件
python -m pytest tests/test_feature_engineering.py

# 运行测试并显示详细输出
python -m pytest -xvs
```

### 测试覆盖率
```bash
# 生成测试覆盖率报告
python -m pytest --cov=src

# 生成HTML格式的覆盖率报告
python -m pytest --cov=src --cov-report=html
```

当前覆盖率：49%
- model.py: 89%
- feature_engineering.py: 61%
- data_loader.py: 71%

## CI/CD
项目已配置 GitHub Actions 工作流，在每次代码提交时自动运行以下任务：
- 代码风格检查
- 依赖安装
- 测试运行
- 测试覆盖率报告生成

工作流配置文件：`.github/workflows/ci.yml`

## 免责声明
示例项目仅用于教学和原型验证，勿直接用于生产决策；真实风控需结合更丰富数据与合规审查。

