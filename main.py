# from ultralytics import YOLO
#
# # 从头开始创建一个新的YOLO模型
# # model = YOLO('./ultralytics/ultralytics/cfg/models/v8/myyolov8.yaml')
# #
# # # 加载预训练的YOLO模型（推荐用于训练）
# # model = YOLO('./yolov8n.pt')
# #
# # # 使用“coco128.yaml”数据集训练模型3个周期
# # results = model.train(data='./ultralytics/ultralytics/cfg/datasets/mycoco128.yaml', epochs=30)
# #
# # # 评估模型在验证集上的性能
# # results = model.val()
#
# infer = YOLO("runs/detect/train10/weights/best.pt")
# results = infer.predict("bus.jpg", save=True)
# # # 使用模型对图片进行目标检测
# # results = model('https://ultralytics.com/images/bus.jpg')
#
# # # 将模型导出为ONNX格式
# # success = model.export(format='onnx')

import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import websockets
import base64
import os
import json

# 加载 YOLO 模型
infer = YOLO("runs/detect/train10/weights/best.pt")
async def handle_image(websocket, path):
    while True:
        try:
            # 接收客户端发送的JSON格式像素多维数组图像数据
            image_json = await websocket.recv()
            # 将JSON格式的多维数组图像数据转为numpy多维数组
            image_data = json.loads(image_json)
            pixels = image_data['pixels']

            img_array = np.array(pixels, dtype=np.uint8)

            # 将一维数组还原为图像的多维数组表示形式
            img = img_array.reshape((image_data['height'], image_data['width'], 3))
            print(img)
            print(img.shape)

            # 进行目标检测
            results = infer.predict(img, show=False, save=True, classes=[0])

            # 获取检测结果的图像路径
            if len(results) > 0 and hasattr(results[0], 'save_dir') and hasattr(results[0], 'path'):
                save_dir = results[0].save_dir
                image_name = results[0].path
                image_path = os.path.join(save_dir, image_name)  # 组合完整路径
                print(f'Save at {save_dir}')
                print(f'Image name is {image_name}')
                print(f'Image path is {image_path}')

                # 读取检测结果的图像
                image_with_boxes = cv2.imread(image_path)

                # 将图像转为 base64 编码的图像数据，并发送给客户端
                _, image_encoded = cv2.imencode('.jpg', image_with_boxes)
                image_bytes = base64.b64encode(image_encoded.tobytes()).decode('utf-8')
                await websocket.send(image_bytes)

                # 显示图像
                cv2.imshow('YOLO Detection', image_with_boxes)
                cv2.waitKey(1)  # 等待1毫秒以确保图像窗口及时更新

            # 响应客户端，可以发送一些信息表示成功处理图像数据
            await websocket.send("Image received and processed successfully")

        except websockets.exceptions.ConnectionClosedError:
            print("Connection closed by the client.")
            break

if __name__ == "__main__":
    # 启动WebSocket服务器，监听在指定的主机和端口上
    start_server = websockets.serve(handle_image, "localhost", 5000)

    # 异步运行服务器
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

