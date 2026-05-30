# 模块: 数据集 YAML 路径校验

目标: 验证数据集 YAML 是否指向正确的 images/labels 目录, 以确保训练能读取标签。

## 最小任务清单
- [x] 定位实际使用的数据集 YAML 文件(如 rgbd_data_mass_1/detect_dataset.yaml)。
- [x] 核对 `path`, `train`, `val`, `test` 是否存在且与真实目录一致。
- [x] 核对 `labels` 与 `images` 是否平级且分割目录一致。
- [x] 记录发现的问题与修正建议(仅记录, 不改 YAML)。
- [x] 若路径无误, 标记通过并进入下一个模块。

## 验证记录
- 使用文件: rgbd_data_mass_1/dataset.yaml
- `path` 指向 C:/git_lib/YOLO_test/ultralytics/rgbd_data_mass_1
- `train/val/test` 已修正为 images/train, images/val, images/test
- labels 目录与 images 平级且含 train/val/test
- 结论: 原因是图像目录命名为 image, 与 Ultralytics 期望的 images 不一致, 已完成修复
