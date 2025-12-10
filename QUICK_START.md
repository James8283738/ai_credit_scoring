# 🚀 快速开始（AI 信贷评分模型）

## 环境准备
```bash
cd "/Users/zhonghuaxiaochu/claude agent/ai_credit_scoring"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式
### 1) 命令行训练与评估
```bash
python main.py \
  --test-size 0.2 \
  --n-estimators 400 \
  --max-depth 4 \
  --learning-rate 0.05
```

### 2) 启动 Streamlit 仪表盘
```bash
streamlit run app.py
```
浏览器打开提示的 URL（通常 http://localhost:8501）。

## 生成报告
命令行或仪表盘完成训练后，会在 `reports/` 目录生成 JSON/PDF，语言支持 `zh`/`de`/`en`：
```bash
python main.py --report-language zh
```

## 上传到 GitHub
```bash
./setup_git.sh
# 或手动：
git init
git add .
git commit -m "init: ai credit scoring"
git remote add origin https://github.com/<your_name>/ai_credit_scoring.git
git push -u origin main
```

## 测试
```bash
pytest
```

## 常见问题
- **数据下载慢/失败**：检查网络，可手动将 German Credit CSV 放入 `data/german_credit.csv`
- **X11/GUI 报错**：本项目使用 Plotly/Streamlit，无需本地图形界面，确保浏览器可用
- **PDF 生成失败**：确认 `reports/` 可写且已安装 `fpdf` 依赖

