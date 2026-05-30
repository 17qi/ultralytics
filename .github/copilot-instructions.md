# AI Development Rules

本项目采用文档驱动的vibecoding流程。

## Context Files
- doc/project-brief.md：项目背景和目标
- doc/data-spec.md：数据格式和路径约定
- doc/preprocessing-rules.md：数据预处理规则
- doc/evaluation-rules.md：评估指标和实验约束
- doc/tasks/progress.md：任务进度
- doc/questions.md：待人工确认问题

## Mandatory Rules
1. 不得猜测用户意图。
2. 不明确的问题必须向我提问且写入doc/questions.md。
3. 未经明确要求，不得修改训练脚本、评估脚本、模型yaml和数据划分文件。
4. 每次只完成一个最小任务。
5. 修改代码前必须说明将修改哪些文件。
6. 修改代码后必须说明diff摘要、运行命令、测试结果和未解决问题。
7. 不得覆盖原始数据。
8. 不得使用测试集选择阈值、模型或超参数。
9. 代码尽量通过ruff、mypy和pytest检查。