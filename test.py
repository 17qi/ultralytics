from ultralytics import YOLO
model = YOLO("runs/detect/train6/weights/best.pt")
results = model.predict(source="YOLODataset/test/images", conf=0.25, save=True)

# 命令行运行
# python detect.py \
#   --weights runs/detect/train/weights/best.pt \
#   --source path/to/your/test/images \
#   --conf 0.25 --save-conf --save-txt


#模型导出（Export）
# yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 simplify=True

#性能评估与基准测试（Benchmarking）
#yolo val model=best.onnx data=.../data.yaml imgsz=640
