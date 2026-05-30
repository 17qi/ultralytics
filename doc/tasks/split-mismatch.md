# 模块: 数据划分一致性

目标: 验证 train/val/test 分割在 images 与 labels 之间一致。

## 最小任务清单
- [ ] 确认 images 与 labels 均含 train/val/test 子目录。
- [ ] 核对 YAML 中的 train/val/test 路径与实际目录一致。
- [ ] 抽样检查同一分割中的 image 与 label 是否配对。
- [ ] 记录任何分割不一致的问题。
