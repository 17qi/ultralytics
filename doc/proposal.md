# 训练结果为零的原因与验证方案

## 现象
- 训练与验证阶段出现: WARNING no labels found in detect set
- Instances 为 0, mAP 为 0

## 当前可确定结论
- 训练/验证阶段未读取到有效检测标签, 因此指标为 0。
- 仍需通过逐项验证来判断是数据缺失还是路径/格式问题。

## 可能原因与逐项验证方案

### 原因 1: 数据集 YAML 指向错误路径
**验证**
- 打开数据集 YAML (例如 rgbd_data_mass_1/detect_dataset.yaml)。
- 检查 `path`, `train`, `val`, `test` 是否指向真实存在的图像目录。
- 若使用相对路径, 确认其相对位置是以 YAML 所在目录为基准。
- 确认 `labels` 目录与 `images` 平级且分割一致。

**解决**
- 修正 YAML 中的路径与分割目录。
- 重新训练以确认标签被正确读取。

### 原因 2: 标签文件缺失或数量为 0
**验证**
- 统计 labels/train, labels/val, labels/test 下的 .txt 数量。
- 对比 images/train, images/val, images/test 的数量。
- 抽样检查是否存在同名配对 (image.png -> label.txt)。

**解决**
- 补全缺失标签或重新导出标注。
- 确保 labels 与 images 的文件名 stem 完全一致。

### 原因 3: 标签文件存在但为空文件
**验证**
- 抽样查看 labels/*.txt, 检查文件是否空行或长度为 0。
- 统计空文件比例。

**解决**
- 重新导出标注, 保证每个目标都写入正确标签行。

### 原因 4: 标签文件命名不匹配
**验证**
- 检查是否存在大小写差异、前后缀不一致或额外字符。
- 对比同一目录下 image 和 label 的 stem 集合是否一致。

**解决**
- 统一命名规则后再训练。

### 原因 5: 标签格式不符合 detect 任务
**验证**
- 每一行格式应为: class x_center y_center width height (归一化到 [0,1])。
- 检查是否误用了 seg/pose 格式或未归一化。
- 检查 class id 是否超出 names 的范围。

**解决**
- 修正标注导出格式, 或改用对应任务的训练入口。

### 原因 6: 数据划分不一致
**验证**
- 确认 images 与 labels 均含 train/val/test 子目录。
- 检查 YAML 中的 train/val/test 是否匹配实际划分目录。

**解决**
- 调整目录结构或 YAML 指向, 使分割一致。

### 原因 7: 使用了过期缓存
**验证**
- 检查数据根目录或 runs 目录中是否存在 .cache 文件。
- 若路径已更新但仍提示无标签, 可能使用了旧缓存。

**解决**
- 删除相关 .cache 文件或在训练命令中禁用缓存。

## 建议验证顺序 (最小成本)
1. 检查数据集 YAML 的 `path/train/val/test` 是否正确。
2. 对比 images 与 labels 的数量和文件名是否一一对应。
3. 抽样查看 labels 内容是否为空且格式正确。
4. 最后排查缓存问题。

## 备注
- 根据现有数据处理计划, 预期结构为:
  - rgbd_data_mass_1/image/{train,val,test}
  - rgbd_data_mass_1/labels/{train,val,test}
- 若实际结构与预期不同, 需优先纠正 YAML 或目录结构。
