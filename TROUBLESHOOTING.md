# GitHub Actions Workflow 故障排除

## 如何查看失败原因

1. 前往 Actions 页面：`https://github.com/eit-nlp-industry/LLaMA-Factory/actions`
2. 点击失败的运行（显示红色 ❌ 的）
3. 点击失败的 job（例如 "Encrypt Repository and Push"）
4. 展开失败的步骤，查看错误日志

## 常见错误及解决方案

### 错误 1: "GPG_PRIVATE_KEY" 未设置或格式错误

**错误信息**：
```
gpg: no valid OpenPGP data found
```

**解决方案**：
1. 检查 GitHub Secrets 中 `GPG_PRIVATE_KEY` 是否已设置
2. 确保私钥格式完整，包含：
   ```
   -----BEGIN PGP PRIVATE KEY BLOCK-----
   ...
   -----END PGP PRIVATE KEY BLOCK-----
   ```
3. 确保没有多余的空格或换行

**验证方法**：
```bash
# 在本地查看私钥格式
cat /home/ziqiang/LLaMA-Factory/.git-crypt-keys/private-key.gpg
```

### 错误 2: "ENCRYPTED_REPO_TOKEN" 权限不足

**错误信息**：
```
remote: Permission denied
fatal: unable to access 'https://...': The requested URL returned error: 403
```

**解决方案**：
1. 检查 `ENCRYPTED_REPO_TOKEN` 是否已设置
2. 确认 token 有 `repo` 权限（完整仓库访问权限）
3. 确认 token 未过期
4. 确认 token 对目标仓库 `zich601/LLaMA-Factory-encrypted` 有访问权限

**重新创建 Token**：
1. 前往 GitHub Settings → Developer settings → Personal access tokens
2. 创建新 token，确保勾选 `repo` 权限
3. 更新 Secret `ENCRYPTED_REPO_TOKEN`

### 错误 3: 目标仓库不存在

**错误信息**：
```
remote: Repository not found
fatal: repository 'https://github.com/zich601/LLaMA-Factory-encrypted.git' not found
```

**解决方案**：
1. 确认加密仓库已创建：`https://github.com/zich601/LLaMA-Factory-encrypted`
2. 确认仓库名称拼写正确（注意大小写）
3. 确认 token 对该仓库有访问权限

### 错误 4: GPG 密钥 ID 获取失败

**错误信息**：
```
GPG_KEY_ID is empty
```

**解决方案**：
1. 检查 GPG 私钥是否正确导入
2. 确认私钥格式正确
3. 检查 workflow 日志中的 GPG 导入步骤

### 错误 5: git-crypt 初始化失败

**错误信息**：
```
git-crypt: error: repository is already initialized
```

**解决方案**：
这是正常的，如果仓库已经初始化过 git-crypt，可以忽略这个错误。workflow 应该继续执行。

## 检查清单

在重新运行 workflow 之前，请确认：

- [ ] `GPG_PRIVATE_KEY` Secret 已设置且格式正确
- [ ] `GPG_PASSPHRASE` Secret 已设置（可以留空）
- [ ] `ENCRYPTED_REPO_TOKEN` Secret 已设置且有效
- [ ] Token 有 `repo` 权限
- [ ] 加密仓库 `zich601/LLaMA-Factory-encrypted` 已创建
- [ ] Token 对加密仓库有访问权限
- [ ] Workflow 文件已正确提交到仓库

## 调试步骤

1. **检查 Secrets 是否设置**：
   - 前往仓库 Settings → Secrets and variables → Actions
   - 确认三个 Secrets 都存在

2. **测试 GPG 密钥**：
   ```bash
   # 在本地测试导入私钥
   cat .git-crypt-keys/private-key.gpg | gpg --batch --yes --import
   gpg --list-secret-keys
   ```

3. **测试 Token 权限**：
   ```bash
   # 使用 token 测试访问
   curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/repos/zich601/LLaMA-Factory-encrypted
   ```

4. **查看详细日志**：
   - 在 Actions 页面点击失败的运行
   - 展开每个步骤查看详细输出

## 重新运行 Workflow

修复问题后：

1. 前往 Actions 页面
2. 点击 `encrypt-and-push.yml`
3. 点击 "Run workflow"
4. 输入参数并运行

如果问题仍然存在，请复制完整的错误日志进行进一步诊断。
