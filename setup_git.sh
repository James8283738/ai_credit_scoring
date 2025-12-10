#!/usr/bin/env bash
# 简易 Git/GitHub 自动化脚本
set -e

REPO_NAME="ai_credit_scoring"

# 检查是否已经是Git仓库
if [ ! -d ".git" ]; then
  echo "初始化 Git 仓库..."
  git init
else
  echo "Git 仓库已存在，跳过初始化..."
fi

# 检查.gitignore是否存在，不存在则创建
if [ ! -f ".gitignore" ]; then
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
else
  echo ".gitignore 已存在，跳过写入..."
fi

# 检查是否有未提交的文件
git status --porcelain | grep -q .
if [ $? -eq 0 ]; then
  echo "添加文件并提交..."
  git add .
  git commit -m "init: ai credit scoring"
else
  echo "没有未提交的文件，跳过提交..."
fi

# 检查是否已经配置了远程仓库
git remote -v | grep -q origin
if [ $? -ne 0 ]; then
  read -p "请输入 GitHub 仓库 URL（例如 https://github.com/you/${REPO_NAME}.git）: " REMOTE_URL
  if [ -n "$REMOTE_URL" ]; then
    git remote add origin "$REMOTE_URL"
    git branch -M main
    
    # 尝试使用HTTP1.1来避免HTTP2协议问题
    git config http.version HTTP/1.1
    
    echo "推送代码到 GitHub..."
    git push -u origin main
    echo "✅ 已推送到 $REMOTE_URL"
  else
    echo "⚠️ 未提供远程地址，已完成本地初始化。"
  fi
else
  echo "远程仓库已配置，跳过添加..."
  echo "如果需要推送最新代码，请使用 'git push' 命令。"
fi

