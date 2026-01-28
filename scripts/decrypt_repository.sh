#!/bin/bash

# 解密仓库脚本
# 用于在部署服务器上解密加密的仓库

set -e

echo "=========================================="
echo "git-crypt 解密脚本"
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

# 检查私钥文件
PRIVATE_KEY_FILE="${1:-.git-crypt-keys/private-key.gpg}"

if [ ! -f "$PRIVATE_KEY_FILE" ]; then
    echo "错误: 私钥文件不存在: $PRIVATE_KEY_FILE"
    echo ""
    echo "使用方法:"
    echo "  $0 [私钥文件路径]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/private-key.gpg"
    echo "  $0 .git-crypt-keys/private-key.gpg"
    exit 1
fi

echo "使用私钥文件: $PRIVATE_KEY_FILE"
echo ""

# 导入 GPG 私钥
echo "正在导入 GPG 私钥..."
gpg --batch --import "$PRIVATE_KEY_FILE"

# 获取 GPG 密钥 ID
GPG_KEY_ID=$(gpg --list-secret-keys --keyid-format LONG | grep -E "sec\s+[a-zA-Z0-9/]+" | head -1 | awk '{print $2}' | cut -d'/' -f2)

if [ -z "$GPG_KEY_ID" ]; then
    echo "错误: 无法获取 GPG 密钥 ID"
    exit 1
fi

echo "GPG 密钥 ID: $GPG_KEY_ID"
echo ""

# 信任 GPG 密钥
echo "正在信任 GPG 密钥..."
echo "$GPG_KEY_ID:6:" | gpg --import-ownertrust

# 解锁 git-crypt
echo "正在解锁 git-crypt..."
echo "请输入 GPG 密码:"
git-crypt unlock

echo ""
echo "=========================================="
echo "解密完成！"
echo "=========================================="
echo ""
echo "仓库已成功解密，您现在可以正常使用所有文件。"
echo ""
