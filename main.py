from ultralytics import YOLO

# 从头开始创建一个新的YOLO模型
# model = YOLO('./ultralytics/ultralytics/cfg/models/v8/myyolov8.yaml')
#
# # 加载预训练的YOLO模型（推荐用于训练）
# model = YOLO('./yolov8n.pt')
#
# # 使用“coco128.yaml”数据集训练模型3个周期
# results = model.train(data='./ultralytics/ultralytics/cfg/datasets/mycoco128.yaml', epochs=30)
#
# # 评估模型在验证集上的性能
# results = model.val()

infer = YOLO("runs/detect/train10/weights/best.pt")
results = infer.predict("bus.jpg", save=True)
# # 使用模型对图片进行目标检测
# results = model('https://ultralytics.com/images/bus.jpg')

# # 将模型导出为ONNX格式
# success = model.export(format='onnx')