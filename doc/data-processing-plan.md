# RGB-D 生菜数据预处理方案

## 目标
为 1280x720 RGB + Depth 数据提供一致、可复现的数据预处理与质检流程，用于 YOLO 生菜 ROI 检测与后续 ROI 深度分析/三维定位。

## 结论（基于采集脚本）
- 采集脚本同时保存原始深度与对齐深度：`depth_raw`(raw) 与 `depth_aligned`(aligned)，并可选保存滤波后深度、深度可视化、深度米制数组与有效深度掩码。
- RGB 与 Depth 的配对规则为同一数值编号的文件名（同一 `stem`），且深度在保存前已对齐到彩色相机坐标系。

参考采集脚本：[data_get_rgbd.py](data_get_rgbd.py)

## 当前数据目录结构（已确认）
- 数据集根目录：`rgbd_data_mass_1`
- `image/{train,val,test}`：RGB 图像
- `labels/{train,val,test}`：YOLO 标签
- `depth_aligned/{train,val,test}`：对齐后的深度 PNG（z16），按划分子目录保存并与 RGB 同名配对
- 已包含可选输出：`depth_raw`、`depth_aligned_filtered`、`depth_vis_*`、`depth_meters` 等

## 关键风险与检查点
1. **RGB-Depth 配对缺失**：RGB 与深度文件编号不一致或缺失。
2. **分割与深度未对齐**：`image/train|val|test` 有划分，但 `depth_aligned` 未按同样划分整理。
3. **固定黑边/无效区**：depth 左侧存在固定无效区域，影响 ROI 深度统计。
4. **无效深度比例过高**：深度值为 0 的像素占比过高。
5. **分辨率与坐标一致性**：RGB/Depth 分辨率不一致或对齐异常。
6. **深度标定信息缺失**：`depth_scale` 来自单独配置，若未归档会导致深度米制化不可追溯。
7. **标签越界**：YOLO 框坐标超出 RGB 边界，影响 ROI 深度统计。

## 推荐数据处理流程（不创建裁剪脚本）
1. **数据清点**
   - 统计 RGB 与 `depth_aligned` 文件数量，并按 `stem` 检查一一对应。
   - 抽样核对 `image/*/*.png` 与 `depth_aligned/*/*.png` 是否同名存在。
2. **对齐一致性确认**
   - 抽样可视化 RGB 与 `depth_aligned` 伪彩色对比，验证像素级对齐。
3. **无效深度与黑边评估**
   - 统计每张深度图中 `depth==0` 的比例。
   - 记录左侧固定黑边的列范围，后续分析按掩码或裁剪忽略该区域。
4. **深度量纲确认**
   - 已有 `depth_meters`，`depth_scale` 来自单独配置。
   - 建议在文档中固化 `depth_scale` 的配置位置与取值。
5. **与 YOLO 标签一致性检查**
   - 验证 `labels/*/*.txt` 与 `image/*/*.png` 的文件名一致。
   - 抽样检查框是否在图像范围内。
6. **输出可追溯清单**
   - 记录：数据版本、样本数量、缺失样本列表、无效深度比例统计、黑边范围（如有）。

## 不在本方案内的事项
- 不修改任何数据文件
- 不创建裁剪脚本
- 不调整模型或训练超参数
