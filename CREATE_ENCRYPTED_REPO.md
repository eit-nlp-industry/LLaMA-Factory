# 创建加密仓库指南

## 当前状态

- **原始仓库**: `eit-nlp-industry/LLaMA-Factory` ✅ 已存在
- **加密仓库**: 需要创建 ⚠️

## 步骤 1: 在 GitHub 上创建新仓库

1. 登录 GitHub，前往组织 `eit-nlp-industry`（或您的个人账户）

2. 点击右上角的 **"+"** → **"New repository"**

3. 填写仓库信息：
   - **Repository name**: `LLaMA-Factory-encrypted`（或您喜欢的名称）
   - **Description**: `Encrypted version of LLaMA-Factory for deployment`
   - **Visibility**: 选择 **Private**（重要！加密仓库应该是私有的）
   - **不要**初始化 README、.gitignore 或 license（这些会由 CI/CD 自动添加）

4. 点击 **"Create repository"**

5. 记录仓库名称，格式为：`owner/repo-name`
   - 例如：`zich601/LLaMA-Factory-encrypted`

## 步骤 2: 在原始仓库中设置 GitHub Secrets

在原始仓库 `eit-nlp-industry/LLaMA-Factory` 中设置：

1. 前往仓库页面：`https://github.com/eit-nlp-industry/LLaMA-Factory`

2. 点击 **Settings** → **Secrets and variables** → **Actions**

3. 点击 **"New repository secret"**，添加以下三个 Secrets：

### Secret 1: GPG_PRIVATE_KEY
- **Name**: `GPG_PRIVATE_KEY`
- **Value**: 复制 `.git-crypt-keys/private-key.gpg` 文件的完整内容
  ```bash
  cat /home/ziqiang/LLaMA-Factory/.git-crypt-keys/private-key.gpg
  ```
  复制从 `-----BEGIN PGP PRIVATE KEY BLOCK-----` 到 `-----END PGP PRIVATE KEY BLOCK-----` 的所有内容

### Secret 2: GPG_PASSPHRASE
- **Name**: `GPG_PASSPHRASE`
- **Value**: 留空（因为使用了无密码密钥）
  - 或者输入一个空字符串 `""`

### Secret 3: ENCRYPTED_REPO_TOKEN
- **Name**: `ENCRYPTED_REPO_TOKEN`
- **Value**: 创建 Personal Access Token
  1. 前往 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
  2. 点击 **"Generate new token (classic)"**
  3. 填写信息：
     - **Note**: `LLaMA-Factory Encryption Token`
     - **Expiration**: 选择适合的期限（建议 90 天或更长）
     - **Scopes**: 勾选 `repo`（完整仓库访问权限）
  4. 点击 **"Generate token"**
  5. **立即复制 token**（只显示一次！）
  6. 在 GitHub 仓库的 Secrets 页面：
     - 点击 **"New repository secret"**
     - **Name** 输入：`ENCRYPTED_REPO_TOKEN`
     - **Secret** 输入：粘贴刚才复制的 Personal Access Token
     - 点击 **"Add secret"**

## 步骤 3: 更新 Workflow 配置

更新 `.github/workflows/encrypt-and-push.yml` 中的默认仓库名称：
 
```yaml
env:
  ENCRYPTED_REPO: ${{ github.event.inputs.target_repo || 'zich601/LLaMA-Factory-encrypted' }}
```

将默认值改为您实际创建的加密仓库名称（格式：`owner/repo-name`）。

## 步骤 4: 测试 Workflow

1. 提交更改：
   ```bash
   git add .github/workflows/encrypt-and-push.yml
   git commit -m "Update encrypted repository name"
   git push
   ```

2. 手动触发 Workflow：
   - **重要**：在**原始仓库**（`eit-nlp-industry/LLaMA-Factory`）的 Actions 页面操作
   - 前往：`https://github.com/eit-nlp-industry/LLaMA-Factory/actions`
   - 在左侧工作流列表中找到：**`encrypt-and-push.yml`**（这是文件名）
   - 点击 **`encrypt-and-push.yml`**
   - 在右侧点击 **"Run workflow"** 下拉按钮
   - 在弹出窗口中：
     - **target_repo**: 输入 `zich601/LLaMA-Factory-encrypted`（或使用默认值）
     - **target_branch**: 输入 `main`（或使用默认值）
   - 点击绿色的 **"Run workflow"** 按钮

   **注意**：如果看到失败的运行，点击查看详细错误信息来诊断问题。

3. 检查结果：
   - 查看 Actions 日志，确认加密和推送成功
   - 前往加密仓库，确认文件已加密（应该是二进制格式）

## 验证加密

在加密仓库中，文件应该是加密的（二进制格式）。只有使用私钥解密后才能看到明文。

## 故障排除

### 问题：Workflow 失败，提示权限错误
- 检查 `ENCRYPTED_REPO_TOKEN` 是否正确设置
- 确认 token 有 `repo` 权限
- 确认加密仓库名称正确

### 问题：GPG 密钥导入失败
- 检查 `GPG_PRIVATE_KEY` 格式是否正确（包含完整的 BEGIN/END 标记）
- 确认没有多余的空格或换行

### 问题：文件没有加密
- 检查 `.gitattributes` 配置
- 确认文件路径匹配加密规则
