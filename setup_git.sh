#!/bin/bash
# Git和GitHub设置脚本 - AI信贷评分模型项目
# 这个脚本会帮助你一步步设置Git并上传代码到GitHub

REPO_NAME="ai_credit_scoring"

echo "=========================================="
echo "  Git 和 GitHub 设置助手"
echo "  项目: AI信贷评分模型"
echo "=========================================="
echo ""

# 检查是否已安装Git
if ! command -v git &> /dev/null; then
    echo "❌ 错误: 未找到Git"
    echo "请先安装Git: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git已安装"
echo ""

# 检查Git配置
echo "检查Git配置..."
if git config --global user.name &> /dev/null && git config --global user.email &> /dev/null; then
    echo "✅ Git已配置"
    echo "  用户名: $(git config --global user.name)"
    echo "  邮箱: $(git config --global user.email)"
else
    echo "⚠️  Git未配置，需要设置用户名和邮箱"
    echo ""
    read -p "请输入你的GitHub用户名: " github_username
    read -p "请输入你的GitHub邮箱: " github_email
    
    git config --global user.name "$github_username"
    git config --global user.email "$github_email"
    
    echo "✅ Git配置完成"
fi

echo ""
echo "=========================================="
echo "  步骤1: 初始化Git仓库"
echo "=========================================="

# 检查是否已初始化
if [ -d ".git" ]; then
    echo "✅ Git仓库已初始化"
else
    echo "正在初始化Git仓库..."
    git init
    echo "✅ Git仓库初始化完成"
fi

echo ""
echo "=========================================="
echo "  步骤2: 配置.gitignore文件"
echo "=========================================="

# 检查.gitignore是否存在，不存在则创建
if [ ! -f ".gitignore" ]; then
  echo "正在创建 .gitignore 文件..."
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
  echo "✅ .gitignore 文件创建完成"
else
  echo "✅ .gitignore 文件已存在"
fi

echo ""
echo "=========================================="
echo "  步骤3: 检查文件状态"
echo "=========================================="

git status

echo ""
echo "=========================================="
echo "  步骤4: 添加文件到Git"
echo "=========================================="

read -p "是否添加所有文件到Git？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add .
    echo "✅ 文件已添加到暂存区"
    git status
else
    echo "跳过添加文件"
    exit 0
fi

echo ""
echo "=========================================="
echo "  步骤5: 提交到本地仓库"
echo "=========================================="

read -p "请输入提交信息（例如：初始提交）: " commit_message
if [ -z "$commit_message" ]; then
    commit_message="初始提交：AI信贷评分模型项目"
fi

git commit -m "$commit_message"
echo "✅ 已提交到本地仓库"

echo ""
echo "=========================================="
echo "  步骤6: 连接GitHub远程仓库"
echo "=========================================="

# 检查是否已有远程仓库
if git remote get-url origin &> /dev/null; then
    echo "✅ 已配置远程仓库:"
    git remote -v
    read -p "是否要更改远程仓库URL？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入新的GitHub仓库URL: " repo_url
        git remote set-url origin "$repo_url"
        echo "✅ 远程仓库URL已更新"
    fi
else
    echo "⚠️  未配置远程仓库"
    echo ""
    echo "请先在GitHub上创建仓库，然后："
    echo "1. 访问 https://github.com/new"
    echo "2. 创建新仓库（不要初始化README）"
    echo "3. 复制仓库URL"
    echo ""
    read -p "请输入GitHub仓库URL（例如：https://github.com/用户名/${REPO_NAME}.git）: " repo_url
    
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✅ 远程仓库已添加"
    else
        echo "⚠️  未输入URL，跳过远程仓库配置"
    fi
fi

echo ""
echo "=========================================="
echo "  步骤7: 推送到GitHub"
echo "=========================================="

if git remote get-url origin &> /dev/null; then
    read -p "是否推送到GitHub？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在推送到GitHub..."
        
        # 尝试使用HTTP1.1来避免HTTP2协议问题
        git config http.version HTTP/1.1
        
        # 检查分支名称
        current_branch=$(git branch --show-current)
        if [ -z "$current_branch" ]; then
            git branch -M main
            current_branch="main"
        fi
        
        git push -u origin "$current_branch"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 成功！代码已上传到GitHub"
            echo ""
            echo "你可以访问以下URL查看你的仓库："
            git remote get-url origin | sed 's/\.git$//'
        else
            echo ""
            echo "❌ 推送失败"
            echo "可能的原因："
            echo "1. 需要输入GitHub用户名和密码（或Personal Access Token）"
            echo "2. 仓库URL不正确"
            echo "3. 网络连接问题"
            echo "4. 可能需要使用SSH而非HTTPS"
            echo ""
            echo "请手动执行: git push -u origin $current_branch"
        fi
    else
        echo "跳过推送"
    fi
else
    echo "⚠️  未配置远程仓库，无法推送"
fi

echo ""
echo "=========================================="
echo "  设置完成！"
echo "=========================================="
echo ""
echo "常用命令："
echo "  git status          - 查看文件状态"
echo "  git add .          - 添加所有文件"
echo "  git commit -m \"消息\" - 提交更改"
echo "  git push            - 推送到GitHub"
echo "  git pull            - 从GitHub拉取更新"
echo ""
echo "项目包含的关键文件："
echo "  main.py            - 主程序入口"
echo "  app.py             - 仪表盘应用"
echo "  config.py          - 配置文件"
echo "  src/               - 核心代码目录"
echo "  tests/             - 测试文件目录"
echo "  .github/workflows/ci.yml - CI/CD配置"

