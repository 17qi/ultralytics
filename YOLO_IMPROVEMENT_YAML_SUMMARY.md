# YOLO 改进版本 YAML 配置对比实验总结

本文档总结了为 YOLOv5、YOLOv8、YOLO12、YOLO26 创建的改进版本 YAML 配置文件，用于系统化的对比实验。

## 改进方向

对标 YOLO11 的改进版本，为其他 YOLO baseline 实现以下三种改进：

1. **P2-DySample**：添加 P2 检测层 + DySample 替代 nn.Upsample
   - 目的：小目标检测能力增强 + 更好的特征上采样

2. **P2-DySample-BiFPN**：P2-DySample + BiFPN 多尺度特征融合
   - 目的：利用双向特征金字塔进行更复杂的特征融合

3. **P2-DySample-SPD1/SPD2**：P2-DySample + Space-to-Depth 下采样
   - P2-DySample-SPD1：替换第1个下采样层（P1→P2）为 SPDConv
   - P2-DySample-SPD2：替换前2个下采样层（P1→P2、P2→P3）为 SPDConv
   - 目的：保留下采样过程中的空间信息，减小特征损失

---

## 文件组织结构

### YOLOv5 改进版本
位置：`ultralytics/cfg/models/v5/`

```
yolov5-P2-DySample.yaml
yolov5-P2-DySample-BiFPN.yaml
yolov5-P2-DySample-SPD1.yaml
yolov5-P2-DySample-SPD2.yaml
```

**特点**：
- 使用 C3 块（YOLOv5 特有）
- 第一层卷积 kernel=6
- head 中 upsample 前带 1×1 Conv

---

### YOLOv8 改进版本
位置：`ultralytics/cfg/models/v8/`

```
yolov8-P2-DySample.yaml
yolov8-P2-DySample-BiFPN.yaml
yolov8-P2-DySample-SPD1.yaml
yolov8-P2-DySample-SPD2.yaml
```

**特点**：
- 使用 C2f 块（YOLOv8 特有）
- 较浅的深度多重因子 (depth_multiple=0.33)
- 相对较小的模型规模

---

### YOLO12 改进版本
位置：`ultralytics/cfg/models/12/`

```
yolo12-P2-DySample.yaml
yolo12-P2-DySample-BiFPN.yaml
yolo12-P2-DySample-SPD1.yaml
yolo12-P2-DySample-SPD2.yaml
```

**特点**：
- 使用 A2C2f 块（YOLO12 特有，带注意力机制）
- 中等深度配置
- 没有 PSA（C2PSA）  结构

---

### YOLO26 改进版本
位置：`ultralytics/cfg/models/26/`

```
yolo26-P2-DySample.yaml
yolo26-P2-DySample-BiFPN.yaml
yolo26-P2-DySample-SPD1.yaml
yolo26-P2-DySample-SPD2.yaml
```

**特点**：
- 使用 C3k2 块（与 YOLO11 相同）
- 包含 C2PSA（路径相关空间注意力）
- 包含 end2end: True 配置
- SPPF 带额外参数：SPPF, [1024, 5, 3, True]

---

## 对比实验建议方案

### 实验1：基础能力对比
```
对比组：
- yolov5.yaml → yolov5-P2-DySample.yaml
- yolov8.yaml → yolov8-P2-DySample.yaml
- yolo12.yaml → yolo12-P2-DySample.yaml
- yolo11.yaml → yolo11-P2-DySample.yaml（已有）

指标：mAP@0.5:0.95、ROI Recall、参数量、FLOPs、推理时间
```

### 实验2：特征融合效果（BiFPN 对比）
```
对比组（以 YOLOv8 为代表）：
- yolov8-P2-DySample.yaml（基础 P2+DySample）
- yolov8-P2-DySample-BiFPN.yaml（加入 BiFPN 融合）

其他版本类似对比
```

### 实验3：下采样方式对比（SPD 变体）
```
对比组（以 YOLOv8 为代表）：
- yolov8-P2-DySample.yaml（标准下采样）
- yolov8-P2-DySample-SPD1.yaml（1层 SPD）
- yolov8-P2-DySample-SPD2.yaml（2层 SPD）

其他版本类似对比
```

### 实验4：跨架构通用性验证
```
相同改进方案应用到不同 baseline 的效果：
- 所有版本应用 P2-DySample 后的 mAP 改进幅度
- SPD 改进在不同架构上的一致性
- BiFPN 在不同架构上的收益变化
```

---

## 快速开始命令

### 训练 YOLOv5-P2-DySample 小目标检测模型
```bash
python train.py --model ultralytics/cfg/models/v5/yolov5-P2-DySample.yaml --data data_3_yolo_mix/weed_lettuce.yaml --epochs 100 --imgsz 640
```

### 训练 YOLOv8-P2-DySample-BiFPN 模型
```bash
python train.py --model ultralytics/cfg/models/v8/yolov8-P2-DySample-BiFPN.yaml --data data_3_yolo_mix/weed_lettuce.yaml --epochs 100 --imgsz 640
```

### 训练 YOLO12 所有 SPD 变体进行对比
```bash
for variant in P2-DySample P2-DySample-SPD1 P2-DySample-SPD2; do
  python train.py --model ultralytics/cfg/models/12/yolo12-${variant}.yaml --data data_3_yolo_mix/weed_lettuce.yaml --epochs 100 --imgsz 640
done
```

### 评估并记录 ROI 指标
```bash
python tools/eval_and_record_roi_metrics.py --model yolov5-P2-DySample.pt
```

---

## 模型架构对应关系

| 版本 | backbone 块 | head 块 | 特殊结构 | 注释 |
|------|-----------|--------|--------|------|
| YOLOv5 | C3 | C3 | 1x1 Conv in head | 较早期架构 |
| YOLOv8 | C2f | C2f | 直接 Upsample | 简化设计 |
| YOLO12 | C3k2/A2C2f | A2C2f | 注意力融合 | 中等复杂度 |
| YOLO11 | C3k2/C2PSA | C3k2 | PSA 注意力 | 改进架构 |
| YOLO26 | C3k2/C2PSA | C3k2 | 完整 PSA+end2end | 最新高性能 |

---

## 关键改进点对标表

| 改进 | 应用版本 | 主要变化 | 预期效果 |
|------|--------|--------|--------|
| **P2 检测层** | v5,v8,v12,v11,v26 | +P2/4 输出 | ROI Recall ↑ |
| **DySample** | 所有 P2 版本 head | nn.Upsample → DySample | 特征质量 ↑ |
| **BiFPN** | P2-DySample-BiFPN | 多尺度加权融合 | 融合能力 ↑ |
| **SPDConv** | P2-DySample-SPD1/2 | stride-2 → SPD+stride-1 | 信息保留 ↑ |

---

## 待验证项目

- [ ] 所有版本 YAML 在 parse_model() 中的正确解析
- [ ] DySample 与 BiFPN 在各版本的参数兼容性
- [ ] SPDConv 对小目标检测的实际改进幅度
- [ ] 不同架构间改进迁移性
- [ ] 训练时间与收敛速度对比

---

## 实验数据记录

建议在训练后在 `weights/roi_eval/model_performance_summary.csv` 中记录：

```csv
model_name,version,p2_dysample,bifpn,spd_level,mAP50,mAP50_95,roi_recall,params,flops,inference_time
yolov5-P2-DySample,v5,true,false,0,...
yolov8-P2-DySample-BiFPN,v8,true,true,0,...
yolo12-P2-DySample-SPD2,v12,true,false,2,...
```

---

**最后更新**：2026年5月7日  
**作者**：自动化实验生成  
**状态**：所有 YAML 配置已生成，等待训练验证
