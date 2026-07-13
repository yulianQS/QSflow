# 为 QSflow 贡献代码

感谢你参与 QSflow。提交代码即表示你同意以项目的
`AGPL-3.0-or-later` 许可证发布你的贡献。

## 本地开发

1. 使用 Python 3.10、3.11 或 3.12。
2. 创建虚拟环境并执行 `pip install -r requirements.txt`。
3. 修改前先创建独立分支。
4. 不要提交数据集、模型权重、API Key、`.avls` 项目状态或个人路径。
5. 提交前执行：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -m unittest discover -s tests -v
python -m compileall -q app.py core ui ai voc tests tools
```

## Pull Request

- 一个 PR 只解决一个主题。
- 说明用户可见变化、验证方式和兼容性影响。
- UI 改动请附截图；状态流改动请补充测试。
- 不要在未经讨论时改变数据格式或删除兼容路径。

## 报告问题

请使用 GitHub Issue 模板，并提供 QSflow 版本、Windows/Python 版本、复现
步骤以及诊断日志中的相关错误。提交日志前请移除隐私信息。
