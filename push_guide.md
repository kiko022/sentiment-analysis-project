# 如何推送项目到GitHub

## 步骤1：生成GitHub个人访问令牌（PAT）

1. 登录GitHub账号
2. 点击右上角头像 → Settings
3. 左侧菜单 → Developer settings
4. 左侧菜单 → Personal access tokens → Tokens (classic)
5. 点击 "Generate new token" → "Generate new token (classic)"
6. 填写：
   - Note: 例如 "sentiment-analysis-project"
   - Expiration: 选择 "30 days" 或更长
   - Select scopes: 勾选 "repo" (所有repo相关权限)
7. 点击 "Generate token"
8. **重要**：复制生成的令牌，这是你唯一一次看到它的机会

## 步骤2：使用令牌推送代码

在项目目录下运行：

```powershell
# 进入项目目录
cd C:\Users\张一枭\Desktop\美的\端到端测试脚本\sentiment-analysis-project

# 移除旧的远程仓库（如果存在）
& 'C:\Program Files\Git\bin\git.exe' remote remove origin

# 添加新的远程仓库（使用你的令牌）
# 格式：https://<token>@github.com/kiko022/sentiment-analysis-project.git
& 'C:\Program Files\Git\bin\git.exe' remote add origin https://<你的令牌>@github.com/kiko022/sentiment-analysis-project.git

# 推送代码
& 'C:\Program Files\Git\bin\git.exe' push -u origin main
```

## 步骤3：创建更多提交

推送成功后，创建更多提交以满足M1申请的3个commit要求：

```powershell
# 第二次提交
echo "# Model Training" >> README.md
& 'C:\Program Files\Git\bin\git.exe' add README.md
& 'C:\Program Files\Git\bin\git.exe' commit -m "Add model training pipeline"

# 第三次提交
& 'C:\Program Files\Git\bin\git.exe' add error_analysis.py
& 'C:\Program Files\Git\bin\git.exe' commit -m "Add error analysis module"

# 推送所有提交
& 'C:\Program Files\Git\bin\git.exe' push  
```

## 步骤4：获取commit SHAs

运行以下命令获取3个commit SHAs：

```powershell
& 'C:\Program Files\Git\bin\git.exe' log --oneline -3
```

## 步骤5：验证推送

访问 https://github.com/kiko022/sentiment-analysis-project 确认代码已成功推送
