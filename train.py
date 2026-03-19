import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO('ultralytics/cfg/models/26/yolo26.yaml')
  # model.load('yolo11n.pt')  #注释则不加载
  results = model.train(
    data='C:/git_lib/YOLO_test/ultralytics/data_3_yolo_mix/weed_lettuce.yaml',  #数据集配置文件的路径
    epochs=100,  #训练轮次总数
    batch=8,  #批量大小，即单次输入多少图片训练
    imgsz=640,  #训练图像尺寸
    workers=0,  #加载数据的工作线程数
    device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='SGD',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
    # close_mosaic_border=False,  # True 删除边界的mosaic图像
    save=True,  # True 保存训练结果
    # resume=False,  # True 恢复训练
    # freeze_layers=0,  # 冻结层数，
)
  
  #命令行方式运行
  # yolo task=detect mode=train model=yolov8n.yaml pretrained=yolov8n.pt data=data.yaml epochs=200 imgsz=640 device=0 workers=8 batch=64 amp=False optimizer='SGD' cache=False


#注意：反斜杠路径在Linux中无效且路径引用方式错误，YOLODataset\data.yaml文件；YOLODataset\data.yaml' does not exist，其中 \` 是Windows路径分隔符，而Ubuntu系统需使用/`。


    #     # Resume training
    # results = model.train(resume=True)

# import zipfile
# try:
#     with zipfile.ZipFile("./model/yolo11n.pt", 'r') as zip_ref:
#         zip_ref.testzip()  # 验证ZIP结构
#     print("ZIP文件结构正常")
# except zipfile.BadZipFile:
#     print("ZIP文件损坏！需重新下载")