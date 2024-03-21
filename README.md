# 一、 摘要
本实验旨在通过分析图像各色通道的平均亮度来确定图像的主要颜色。假设是主要色通道将相比其他通道有更高的平均亮度。实验使用了两个常见的图像处理库OpenCV和PIL来读取图像并计算平均亮度。

# 二、 引言
图像处理是计算机视觉的关键方面。确定图像中物体的颜色是可以通过各种方法解决的基础任务。在本实验中，我们分析不同颜色通道的平均亮度来推断图像的颜色。

# 三、材料与方法
使用智能手机摄像头获取了三张纯色物体的图像（红色、蓝色和绿色），它们作为分析的测试对象。然后这些图像被加载到Python环境中，并转换为张量进行处理。

实验中同时使用了OpenCV和PIL库来读取和处理图像数据。对每个图像张量，应用mean函数来计算每个颜色通道（红色、绿色和蓝色）的平均亮度。

图像处理的步骤如下：

1.使用OpenCV和PIL库读取每张图像。
2.将图像转换为张量格式。
3.使用mean()函数计算每个颜色通道的平均亮度。


# 四、结果
使用OpenCV和PIL得到的每张图像的平均亮度值如下：

红色图像
  OpenCV平均亮度: [0.8098, 0.2547, 0.2488]
  PIL平均亮度: [0.8098, 0.2547, 0.2488]

蓝色图像
  OpenCV平均亮度: [0.4152, 0.5833, 0.6700]
  PIL平均亮度: [0.4152, 0.5833, 0.6700]

绿色图像
  OpenCV平均亮度: [0.0575, 0.5376, 0.0319]
  PIL平均亮度: [0.0575, 0.5376, 0.0319]

可以看到不同颜色图像在不同通道显示出了很高的数值，人工可以通过平均亮度直接判断出图像的颜色。

根据平均亮度值，每幅图像的主导颜色被确定如下：

红色图像在红色通道有最高的平均亮度。

蓝色图像在蓝色通道有最高的平均亮度。

绿色图像在绿色通道有最高的平均亮度。

这些结果支持我们的假设，即平均亮度最高的通道指示了图像的主导颜色。

有趣的是，使用OpenCV和PIL得到的亮度值之间没有显著差异，表明这两个库在处理颜色信息方面方式相似。

# 五、 代码

## 1 导入库

`import cv2
from PIL import Image
import torch
import numpy as np`


## 2  定义图片路径
`red_pic = 'pic/red.jpg'
blue_pic = 'pic/blue.jpg'
green_pic = 'pic/green_1.jpg'`

## 3  使用OpenCV读取图片
`def read_image_opencv(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    return torch.tensor(image).permute(2, 0, 1) / 255.0  # 转换为张量并归一化`

## 4  使用PIL读取图片
`def read_image_pil(filepath):
    image = Image.open(filepath)
    image = np.array(image)
    return torch.tensor(image).permute(2, 0, 1) / 255.0  # 转换为张量并归一化`

## 5 mean()计算图像各个通道的平均亮度
`def calculate_mean_brightness(tensor):
    return tensor.mean(dim=(1, 2))`


## 6 主函数
`for filepath in [red_pic, blue_pic, green_pic]:
    color = "Unknown"
    if "red" in filepath:
        color = "Red"
    elif "blue" in filepath:
        color = "Blue"
    elif "green" in filepath:
        color = "Green"`

    tensor_opencv = read_image_opencv(filepath)
    tensor_pil = read_image_pil(filepath)

    mean_brightness_opencv = calculate_mean_brightness(tensor_opencv)
    mean_brightness_pil = calculate_mean_brightness(tensor_pil)

    print(f"{color} Image Average brightness (OpenCV): {mean_brightness_opencv}")
    print(f"{color} Image Average brightness (PIL): {mean_brightness_pil}")

# 2024_spring_avg_brightness
