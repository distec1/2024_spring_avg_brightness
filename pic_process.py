import matplotlib

import cv2
from PIL import Image
import torch
import numpy as np

# 定义图片路径
red_pic = 'pic/red.jpg'
blue_pic = 'pic/blue.jpg'
green_pic = 'pic/green_1.jpg'


# 使用OpenCV读取图片
def read_image_opencv(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    return torch.tensor(image).permute(2, 0, 1) / 255.0  # 转换为张量并归一化


# 使用PIL读取图片
def read_image_pil(filepath):
    image = Image.open(filepath)
    image = np.array(image)
    return torch.tensor(image).permute(2, 0, 1) / 255.0  # 转换为张量并归一化


# 计算图像各个通道的平均亮度
def calculate_mean_brightness(tensor):
    return tensor.mean(dim=(1, 2))


for filepath in [red_pic, blue_pic, green_pic]:
    color = "Unknown"
    if "red" in filepath:
        color = "Red"
    elif "blue" in filepath:
        color = "Blue"
    elif "green" in filepath:
        color = "Green"

    tensor_opencv = read_image_opencv(filepath)
    tensor_pil = read_image_pil(filepath)

    mean_brightness_opencv = calculate_mean_brightness(tensor_opencv)
    mean_brightness_pil = calculate_mean_brightness(tensor_pil)

    print(f"{color} Image Average brightness (OpenCV): {mean_brightness_opencv}")
    print(f"{color} Image Average brightness (PIL): {mean_brightness_pil}")






