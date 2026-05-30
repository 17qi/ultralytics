# 模块: 标签缺失与数量检查

目标: 验证 labels 文件是否缺失或数量为 0。

## 最小任务清单
- [x] 统计 labels/train, labels/val, labels/test 的 .txt 数量。
- [x] 统计 images/train, images/val, images/test 的图像数量。
- [x] 对比同一分割下 images 与 labels 数量是否一致。
- [x] 抽样核对同名配对是否存在 (image.png -> label.txt)。
- [x] 记录缺失列表与比例(如有)。

## 验证记录
- images: train 475, val 136, test 68
- labels: train 476, val 137, test 69
- 差异来自 labels 各分割存在 classes.txt (额外 1 个)
- 结论: 未发现 labels 缺失, 但存在额外文件
