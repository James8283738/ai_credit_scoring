#!/usr/bin/env bash
# 简易 Git/GitHub 自动化脚本
set -e

REPO_NAME="ai_credit_scoring"

echo "初始化 Git 仓库..."
git init

echo "写入 .gitignore..."
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.DS_Store
models/
reports/*.pdf
reports/*.json
data/*.csv
EOF

echo "添加文件并提交..."
git add .
git commit -m "init: ai credit scoring"

read -p "请输入 GitHub 仓库 URL（例如 https://github.com/you/${REPO_NAME}.git）: " REMOTE_URL
if [ -n "$REMOTE_URL" ]; then
  git remote add origin "$REMOTE_URL"
  git branch -M main
  git push -u origin main
  echo "✅ 已推送到 $REMOTE_URL"
else
  echo "⚠️ 未提供远程地址，已完成本地初始化。"
fi

