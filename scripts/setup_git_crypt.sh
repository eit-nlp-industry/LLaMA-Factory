#!/bin/bash

# git-crypt 初始化脚本
# 用于设置 GPG 密钥和初始化 git-crypt

set -e

echo "=========================================="
echo "git-crypt 初始化脚本"
echo "=========================================="

# 检查 git-crypt 是否已安装
if ! command -v git-crypt &> /dev/null; then
    echo "错误: git-crypt 未安装"
    echo "请先安装 git-crypt:"
    echo "  Ubuntu/Debian: sudo apt-get install git-crypt"
    echo "  macOS: brew install git-crypt"
    echo "  CentOS/RHEL: sudo yum install git-crypt"
    exit 1
fi

# 检查 GPG 是否已安装
if ! command -v gpg &> /dev/null; then
    echo "错误: GPG 未安装"
    echo "请先安装 GPG:"
    echo "  Ubuntu/Debian: sudo apt-get install gnupg"
    echo "  macOS: brew install gnupg"
    exit 1
fi

# 检查是否已有 GPG 密钥
if [ -z "$(gpg --list-secret-keys --keyid-format LONG 2>/dev/null)" ]; then
    echo "未检测到 GPG 密钥，正在生成新的 GPG 密钥对..."
    echo ""
    
    # 清理可能的锁文件和 agent
    rm -f ~/.gnupg/.#lk* 2>/dev/null
    gpgconf --kill gpg-agent 2>/dev/null || true
    sleep 1
    
    # 使用批处理模式生成密钥（避免权限问题）
    echo "请输入以下信息:"
    read -p "姓名 (Real name) [默认: LLaMA-Factory Deploy]: " name
    name=${name:-"LLaMA-Factory Deploy"}
    
    read -p "邮箱 (Email) [默认: deploy@llamafactory.local]: " email
    email=${email:-"deploy@llamafactory.local"}
    
    echo ""
    echo "使用配置:"
    echo "  姓名: $name"
    echo "  邮箱: $email"
    echo "  密钥大小: 4096 bits"
    echo "  有效期: 永不过期"
    echo ""
    
    # 创建临时配置文件
    TEMP_CONFIG=$(mktemp)
    cat > "$TEMP_CONFIG" <<EOF
%no-protection
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: $name
Name-Email: $email
Expire-Date: 0
EOF
    
    echo "正在生成密钥（这可能需要几分钟，请耐心等待）..."
    # 使用批处理模式，避免 agent 权限问题
    gpg --batch --generate-key "$TEMP_CONFIG" 2>&1
    
    # 清理临时文件
    rm -f "$TEMP_CONFIG"
    
    echo ""
    echo "GPG 密钥生成完成！"
else
    echo "检测到已存在的 GPG 密钥:"
    gpg --list-secret-keys --keyid-format LONG
    echo ""
    read -p "是否使用现有密钥？(y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        echo "正在生成新的 GPG 密钥对..."
        
        # 清理可能的锁文件和 agent
        rm -f ~/.gnupg/.#lk* 2>/dev/null
        gpgconf --kill gpg-agent 2>/dev/null || true
        sleep 1
        
        read -p "姓名 (Real name) [默认: LLaMA-Factory Deploy]: " name
        name=${name:-"LLaMA-Factory Deploy"}
        
        read -p "邮箱 (Email) [默认: deploy@llamafactory.local]: " email
        email=${email:-"deploy@llamafactory.local"}
        
        TEMP_CONFIG=$(mktemp)
        cat > "$TEMP_CONFIG" <<EOF
%no-protection
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: $name
Name-Email: $email
Expire-Date: 0
EOF
        
        echo "正在生成密钥..."
        gpg --batch --generate-key "$TEMP_CONFIG" 2>&1
        rm -f "$TEMP_CONFIG"
    fi
fi

# 获取 GPG 密钥 ID
GPG_KEY_ID=$(gpg --list-secret-keys --keyid-format LONG | grep -E "sec\s+[a-zA-Z0-9/]+" | head -1 | awk '{print $2}' | cut -d'/' -f2)

if [ -z "$GPG_KEY_ID" ]; then
    echo "错误: 无法获取 GPG 密钥 ID"
    exit 1
fi

echo ""
echo "GPG 密钥 ID: $GPG_KEY_ID"
echo ""

# 初始化 git-crypt
echo "正在初始化 git-crypt..."
git-crypt init

# 添加 GPG 用户
echo "正在添加 GPG 用户到 git-crypt..."
# 使用 --no-commit 避免自动提交，我们稍后手动处理
git-crypt add-gpg-user --no-commit "$GPG_KEY_ID" 2>&1 || {
    # 如果失败，尝试不使用 --no-commit
    echo "重试添加 GPG 用户..."
    git-crypt add-gpg-user "$GPG_KEY_ID" 2>&1 || {
        echo "警告: 添加 GPG 用户时出现问题，但可能已经完成"
    }
}

# 导出公钥和私钥
echo ""
echo "正在导出 GPG 密钥..."
mkdir -p .git-crypt-keys

# 导出公钥
gpg --armor --export "$GPG_KEY_ID" > .git-crypt-keys/public-key.gpg
echo "公钥已保存到: .git-crypt-keys/public-key.gpg"

# 导出私钥（批处理模式，不需要密码）
echo ""
echo "正在导出私钥..."
gpg --batch --yes --armor --export-secret-keys "$GPG_KEY_ID" > .git-crypt-keys/private-key.gpg
echo "私钥已保存到: .git-crypt-keys/private-key.gpg"

# 显示密钥信息
echo ""
echo "=========================================="
echo "设置完成！"
echo "=========================================="
echo ""
echo "重要信息:"
echo "  - GPG 密钥 ID: $GPG_KEY_ID"
echo "  - 公钥文件: .git-crypt-keys/public-key.gpg"
echo "  - 私钥文件: .git-crypt-keys/private-key.gpg"
echo ""
echo "下一步操作:"
echo "  1. 将 .git-crypt-keys/private-key.gpg 添加到 GitHub Secrets 作为 GPG_PRIVATE_KEY"
echo "  2. 由于使用了无密码密钥，GPG_PASSPHRASE 可以留空或设置为空字符串"
echo "  3. 在目标加密仓库创建 Personal Access Token，添加到 GitHub Secrets 作为 ENCRYPTED_REPO_TOKEN"
echo "  4. 更新 .github/workflows/encrypt-and-push.yml 中的 ENCRYPTED_REPO 默认值"
echo ""
echo "注意: 请妥善保管私钥文件，不要将其提交到仓库！"
echo ""
