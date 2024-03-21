#from CSDN: https://blog.csdn.net/weixin_44299786/article/details/132155491



# 利用PIL库加载图像，并查看图像形状以及通道数
import numpy as np
from PIL import Image

# Load the image
image_path = r"pic/blue.jpg"

image = Image.open(image_path)
image_np = np.array(image)  # Convert PIL image to NumPy array

# Get image dimensions
print(image.size)

# Check the number of channels
num_channels = len(image.getbands())

# Determine if it's grayscale or RGB
if num_channels == 1:
    print("The image is grayscale.")
elif num_channels == 3:
    print("The image is RGB.")
else:
    print("The image has", num_channels, "channels, which might indicate an RGBA image or some other format.")

# 利用OpenCV库加载图像，并查看图像的通道数
import cv2



print('    ')



# Load the image
image_path = r"pic/blue.jpg"

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 加载为未修改的图像 # 使用 OpenCV 的 cv2.imread(img_file_path) 加载图像
# image = cv2.imread(image_path)                          # 若不加cv2.IMREAD_UNCHANGED该语句，此时原图像虽然为四通道图像，但是会默认为加载三通道图像

image = cv2.resize(image, (224, 224))  # 将图像调整为指定的大小 resize
print("Image shape:", image.shape)  # 打印图像的形状及通道数
