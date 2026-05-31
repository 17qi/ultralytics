# 待确认问题

1. 受项目约束仅允许在 doc/ 下创建或修改 Markdown 文件。是否可以将输出文档放在 doc/proposal.md（或 doc/solutions/proposal.md）而不是 doc1/proposal.md？
	- 已确认：允许输出到 doc/proposal.md。
2. doc/detailed-design.md 未找到。请确认详细设计文件的实际路径或是否需要我先创建该文件。
    - 已确认：未创建 doc/detailed-design.md，直接使用doc/proposal.md 来指导模块划分。
3. 模块划分是否按 doc/proposal.md 中的 7 个“原因”作为 7 个模块？若不是, 请提供模块列表与命名规则。
是

4. RGB-D 同步裁剪脚本的输出目录命名与位置是什么？例如是否为 rgbd_data_mass_1/images_crop 与 rgbd_data_mass_1/depth_aligned_crop，或由命令行参数指定。
5. 异常日志的具体路径与文件名要求是什么？是否统一写入 reports/ 目录（例如 reports/crop_errors.log）。
6. 输入/输出文件格式是否固定为 .png？深度图是否始终为 16-bit PNG（z16），并需使用 cv2.IMREAD_UNCHANGED 保留原始类型？
7. 当 crop_config.yaml 中的裁剪参数与 source_width/height 或 output_width/height 不一致时，是否应直接报错并退出？
8. 发现 RGB 与 depth 缺失配对或尺寸不一致时，脚本应继续处理其它样本还是立即失败？
9. --dry-run 的预期输出是什么（仅打印计划与统计，还是也写入日志文件）。
10. 是否仅处理 images/{train,val,test} 与 depth_aligned/{train,val,test}，还是需要支持其它子集目录（如空目录或额外分区）。

4=RGB-D 同步裁剪脚本的输出目录命名为“data_processed”,位置固定在 rgbd_data_mass_1 下；
5= 异常日志的输出目录命名为“reports”,位置固定在 rgbd_data_mass_1 下；
6= 输入/输出文件格式固定为 .png, 深度图为 16-bit PNG（z16）保持不变，并需使用 cv2.IMREAD_UNCHANGED 读取原始类型并保存原始类型；
7= 当 crop_config.yaml 中的裁剪参数与 source_width/height 或 output_width/height 不一致时，脚本应直接报错并退出；
8= 缺失配对或尺寸不一致时，脚本应继续处理其它样本，而不是立即失败,同时将错误位置放入异常日志；
9= --dry-run 的预期输出是仅打印计划与统计，但不写入日志文件；
10= 仅处理 images/{train,val,test} 与 depth_aligned/{train,val,test}，不支持其它子集目录。
