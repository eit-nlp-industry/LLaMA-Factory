# 快速开始：仓库加密设置

## ✅ 方案可行性

您提出的方案**完全可行**！使用 git-crypt + GPG 密钥对可以实现：
- ✅ 代码加密存储
- ✅ 只有拥有私钥的人才能解密
- ✅ CI/CD 自动加密并推送到新仓库
- ✅ 在部署服务器上使用私钥解密

## 🚀 快速开始（5 步）

### 步骤 1: 本地初始化 git-crypt

```bash
cd /home/ziqiang/LLaMA-Factory
chmod +x scripts/setup_git_crypt.sh
./scripts/setup_git_crypt.sh
```

这会生成 GPG 密钥对并初始化 git-crypt。

### 步骤 2: 配置 GitHub Secrets

在您的原始仓库中，前往 `Settings → Secrets and variables → Actions`，添加：

1. **GPG_PRIVATE_KEY**: 复制 `.git-crypt-keys/private-key.gpg` 的内容
2. **GPG_PASSPHRASE**: 输入您设置的 GPG 密码
3. **ENCRYPTED_REPO_TOKEN**: 创建 Personal Access Token（需要 `repo` 权限）

### 步骤 3: 创建加密仓库

在 GitHub 上创建新仓库（例如：`your-org/llamafactory-encrypted`），设置为私有。

### 步骤 4: 更新 Workflow 配置

编辑 `.github/workflows/encrypt-and-push.yml`，将第 9 行的默认仓库名称改为您的加密仓库：

```yaml
ENCRYPTED_REPO: ${{ github.event.inputs.target_repo || 'your-org/llamafactory-encrypted' }}
```

### 步骤 5: 提交并触发

```bash
git add .
git commit -m "Add git-crypt encryption setup"
git push
```

然后前往 GitHub Actions 页面手动触发 workflow，或等待自动触发（当推送到 main/master 分支时）。

## 📦 在部署服务器上解密

```bash
# 1. 安装 git-crypt
sudo apt-get install git-crypt gnupg

# 2. 克隆加密仓库
git clone https://github.com/your-org/llamafactory-encrypted.git
cd llamafactory-encrypted

# 3. 解密（需要私钥文件和密码）
chmod +x scripts/decrypt_repository.sh
./scripts/decrypt_repository.sh /path/to/private-key.gpg
```

## 📋 加密范围

根据 `.gitattributes` 配置，以下内容会被加密：
- ✅ `src/` 目录（核心代码）
- ✅ 配置文件（`.yaml`, `.yml`）
- ✅ Python 脚本
- ✅ 数据文件（`data/`, `saves/`, `output/`）
- ✅ 敏感文档

以下内容**不会**加密：
- ❌ README 文件
- ❌ LICENSE
- ❌ `.github/` (CI/CD 配置)
- ❌ `examples/` (示例文件)
- ❌ `tests/` (测试文件)

## 🔒 安全提示

1. **私钥安全**: 永远不要提交私钥到仓库
2. **密码管理**: 使用强密码并妥善保管
3. **访问控制**: 加密仓库应设为私有
4. **定期备份**: 备份私钥到安全位置

## 📚 详细文档

查看 `GIT_CRYPT_SETUP.md` 获取完整的使用指南和故障排除。

## ❓ 常见问题

**Q: 如果忘记 GPG 密码怎么办？**
A: 无法恢复，需要重新生成密钥对并重新加密。

**Q: 可以添加多个用户吗？**
A: 可以，使用 `git-crypt add-gpg-user KEY_ID` 添加更多用户。

**Q: 加密会影响性能吗？**
A: 影响很小，git-crypt 使用透明加密，只在提交/检出时加解密。
