# Git-Crypt 加密仓库设置指南

本指南将帮助您设置 git-crypt 来加密仓库内容，并通过 CI/CD 自动推送到加密仓库。

## 方案概述

这个方案使用 **git-crypt** 和 **GPG 密钥对**来实现：
- ✅ 源代码加密存储
- ✅ 只有拥有私钥的人才能解密
- ✅ 通过 CI/CD 自动加密并推送到新仓库
- ✅ 在部署服务器上使用私钥解密

## 工作流程

```
原始仓库 (明文) 
    ↓
GitHub Actions (自动加密)
    ↓
加密仓库 (密文) 
    ↓
部署服务器 (使用私钥解密)
    ↓
可用的明文代码
```

## 第一步：本地设置 git-crypt

### 1. 安装 git-crypt

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install git-crypt
```

**macOS:**
```bash
brew install git-crypt
```

**CentOS/RHEL:**
```bash
sudo yum install git-crypt
```

### 2. 运行初始化脚本

```bash
chmod +x scripts/setup_git_crypt.sh
./scripts/setup_git_crypt.sh
```

这个脚本会：
- 检查并安装依赖
- 生成 GPG 密钥对（如果不存在）
- 初始化 git-crypt
- 导出公钥和私钥到 `.git-crypt-keys/` 目录

### 3. 查看加密规则

加密规则定义在 `.gitattributes` 文件中。默认情况下，以下内容会被加密：
- `src/` 目录（核心代码）
- 配置文件（`.yaml`, `.yml`，部分排除）
- Python 脚本（部分排除）
- 数据文件（`data/`, `saves/`, `output/`）
- 敏感文档

您可以根据需要修改 `.gitattributes` 文件来调整加密范围。

## 第二步：配置 GitHub Secrets

### 1. 获取 GPG 私钥

私钥文件位于 `.git-crypt-keys/private-key.gpg`。将其内容复制：

```bash
cat .git-crypt-keys/private-key.gpg
```

### 2. 获取 GPG 密码

这是您在生成 GPG 密钥时设置的密码。

### 3. 创建 Personal Access Token

1. 前往 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 创建新 token，需要 `repo` 权限
3. 复制 token 值

### 4. 在原始仓库中添加 Secrets

前往您的原始仓库：`Settings → Secrets and variables → Actions`

添加以下 Secrets：

| Secret 名称 | 值 | 说明 |
|------------|-----|------|
| `GPG_PRIVATE_KEY` | `.git-crypt-keys/private-key.gpg` 的内容 | GPG 私钥 |
| `GPG_PASSPHRASE` | 您的 GPG 密码 | GPG 密钥密码 |
| `ENCRYPTED_REPO_TOKEN` | Personal Access Token | 用于推送到加密仓库的 token |

## 第三步：创建加密仓库

1. 在 GitHub 上创建一个新仓库（例如：`your-org/llamafactory-encrypted`）
2. 确保该仓库是私有的（推荐）
3. 记录仓库名称（格式：`owner/repo`）

## 第四步：配置 GitHub Actions

### 1. 更新 workflow 配置

编辑 `.github/workflows/encrypt-and-push.yml`，更新默认的加密仓库名称：

```yaml
env:
  ENCRYPTED_REPO: ${{ github.event.inputs.target_repo || 'your-org/llamafactory-encrypted' }}
```

将 `your-org/llamafactory-encrypted` 替换为您的实际加密仓库名称。

### 2. 测试 workflow

1. 提交所有更改：
   ```bash
   git add .
   git commit -m "Add git-crypt encryption setup"
   git push
   ```

2. 手动触发 workflow：
   - 前往 GitHub Actions 页面
   - 选择 "Encrypt and Push to Encrypted Repository"
   - 点击 "Run workflow"
   - 输入目标仓库名称和分支

## 第五步：在部署服务器上解密

### 1. 安装 git-crypt

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install git-crypt gnupg

# CentOS/RHEL
sudo yum install git-crypt gnupg
```

### 2. 克隆加密仓库

```bash
git clone https://github.com/your-org/llamafactory-encrypted.git
cd llamafactory-encrypted
```

### 3. 解密仓库

将私钥文件传输到服务器（使用安全的方式，如 SSH），然后运行：

```bash
chmod +x scripts/decrypt_repository.sh
./scripts/decrypt_repository.sh /path/to/private-key.gpg
```

输入 GPG 密码后，仓库将被解密。

### 4. 验证解密

```bash
# 检查文件是否已解密
ls -la src/
cat src/llamafactory/train/sft/trainer.py | head -20
```

如果能看到文件内容（而不是加密的二进制数据），说明解密成功。

## 安全注意事项

1. **私钥安全**
   - 永远不要将私钥提交到任何仓库
   - 使用安全的方式传输私钥到部署服务器
   - 考虑使用密钥管理服务（如 AWS Secrets Manager, HashiCorp Vault）

2. **密码管理**
   - 使用强密码作为 GPG 密码
   - 将密码存储在安全的密码管理器中
   - 不要将密码硬编码在任何脚本中

3. **访问控制**
   - 加密仓库应该是私有的
   - 限制对 GitHub Secrets 的访问
   - 定期轮换 GPG 密钥和密码

4. **备份**
   - 备份私钥文件到安全的位置
   - 记录 GPG 密钥 ID 和密码（存储在安全的地方）

## 故障排除

### 问题：git-crypt unlock 失败

**解决方案：**
- 确保私钥已正确导入：`gpg --list-secret-keys`
- 检查密钥 ID 是否匹配
- 确保输入了正确的密码

### 问题：GitHub Actions 加密失败

**解决方案：**
- 检查 GitHub Secrets 是否正确设置
- 验证 GPG_PRIVATE_KEY 格式是否正确（应该包含 `-----BEGIN PGP PRIVATE KEY BLOCK-----`）
- 检查 ENCRYPTED_REPO_TOKEN 是否有正确的权限

### 问题：某些文件没有被加密

**解决方案：**
- 检查 `.gitattributes` 文件中的规则
- 确保文件没有被 `.gitignore` 忽略
- 运行 `git-crypt status` 查看加密状态

## 常用命令

```bash
# 查看加密状态
git-crypt status

# 锁定（加密）仓库
git-crypt lock

# 解锁（解密）仓库
git-crypt unlock

# 查看 GPG 密钥
gpg --list-secret-keys --keyid-format LONG

# 导出公钥
gpg --armor --export KEY_ID > public-key.gpg

# 导出私钥
gpg --armor --export-secret-keys KEY_ID > private-key.gpg
```

## 参考资源

- [git-crypt 官方文档](https://www.agwa.name/projects/git-crypt/)
- [GPG 使用指南](https://www.gnupg.org/documentation/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
