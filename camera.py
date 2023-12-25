import cv2
import os
from ultralytics import YOLO

print("[INFO] loading YOLO from disk...")  # 可以打印下信息

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


cameraCapture = cv2.VideoCapture(0)  # 打开编号为0的摄像头
infer = YOLO("runs/detect/train10/weights/best.pt") # 加载训练结果最好的模型

cv2.namedWindow('YOLO Detection')  # 创建一个窗口
cv2.setMouseCallback('YOLO Detection', onMouse)  # 设置鼠标回调函数

success, frame = cameraCapture.read()  # 读取摄像头的当前帧
print(f"Frame: {frame}")
print(f"Frame shape: {frame.shape}")
while success and cv2.waitKey(1) == -1 and not clicked:  # 当循环没结束，并且剩余的帧数大于零时进行下面的程序
    results = infer.predict(frame,
                            show=False,# 如果可能，显示出来（True时），YOLO内置函数
                            save=True,# 保存图片
                            classes=[0]# 只检测人
                            )
    # 用读取的方式显示，图片路径不变，内容实时更新
    if len(results) > 0 and hasattr(results[0], 'save_dir') and hasattr(results[0], 'path'):
        save_dir = results[0].save_dir
        image_name = results[0].path
        image_path = os.path.join(save_dir, image_name)  # 组合完整路径

        image = cv2.imread(image_path)
        if image is not None:
            cv2.imshow('YOLO Detection', image)
        else:
            print("Error: 无法加载图像。")
    if cv2.waitKey(1) == -1:
        success, frame = cameraCapture.read()  # 摄像头获取下一帧
    else:
        clicked = True  # 鼠标点击窗口，窗口关闭
cv2.destroyAllWindows()
cameraCapture.release()