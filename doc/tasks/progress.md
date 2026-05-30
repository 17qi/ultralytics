# 任务进度

- 2026-05-30: 新增 RGB-D 数据集审查工具与评估脚本说明（脚本、测试、报告输出），生成 reports/pair_check.csv、reports/depth_quality_summary.csv 与 20 张叠加图。

## 评估脚本说明

### 1) scripts/audit_rgbd_dataset.py
**作用**
- 对 RGB 与 Depth 数据做配对审查（按文件名或时间戳推导的配对信息）。
- 检查 RGB/Depth 分辨率一致性。
- 统计每张 depth 图中 depth==0 的比例。
- 统计每一列像素在全体 depth 图中的无效比例。
- 给出左右无效边界的候选范围（不执行裁剪）。
- 随机抽取 20 组 RGB+Depth 生成叠加可视化。
- 校验 depth 图是否以 16-bit 方式读取（避免被错误当作 8-bit）。

**输入**
- 预处理方案文档：doc/data-processing-plan.md
- RGB 根目录：C:\git_lib\YOLO_test\ultralytics\rgbd_data_mass_1\image
- Depth 根目录：C:\git_lib\YOLO_test\ultralytics\rgbd_data_mass_1\depth_aligned

**输出**
- reports/pair_check.csv
- reports/depth_quality_summary.csv
- reports/overlay_samples/（20 张叠加图）

**操作说明**
1. 确认 RGB 与 depth 目录存在且包含 .png。
2. 在仓库根目录执行：

```bash
python scripts/audit_rgbd_dataset.py --plan doc/data-processing-plan.md --rgb-dir "C:\\git_lib\\YOLO_test\\ultralytics\\rgbd_data_mass_1\\image" --depth-dir "C:\\git_lib\\YOLO_test\\ultralytics\\rgbd_data_mass_1\\depth_aligned" --reports-dir reports
```

3. 查看输出 CSV 与叠加图目录。

### 2) tests/test_audit_rgbd_dataset.py
**作用**
- 对审查脚本的核心输出进行最小单元测试（报告文件是否生成、叠加图是否输出）。

**操作说明**
1. 在仓库根目录执行：

```bash
python -m pytest tests/test_audit_rgbd_dataset.py
```

2. 若 pytest 报错缺少 `cv2`，需先安装 OpenCV 或改用已有环境。

## 本次分析结论（基于 reports 输出）
- 配对：总计 679 组 RGB-Depth，配对成功 679，缺失 0，配对率 100%。
- 分辨率：RGB 与 depth 无分辨率不一致记录（0 条 mismatch）。
- depth 数据类型：全部为 uint16，未发现 8-bit 误读。
- depth==0 占比：均值 0.1211，中位数 0.1166，P95 0.1598，范围 0.0803-0.2263。
- 列无效比例：均值 0.1211，中位数 0.0359，P95 0.9928，最大值 1.0。
- 无效边界候选：左侧无效列结束位置为 10（阈值 0.990）；右侧未检测到稳定无效区（-1）。
- 叠加可视化：已生成 20 张，位于 reports/overlay_samples。

## 模块进度
- [x] yaml-path
- [x] missing-labels
- [x] empty-labels
- [ ] name-mismatch
- [ ] label-format
- [ ] split-mismatch
- [x] cache-stale
