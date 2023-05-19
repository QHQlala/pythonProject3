import cv2
import numpy as np


def extract_red_pixels(image):
    """提取图像中的红色像素"""
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置红色的HSV范围
    lower_red = np.array([0, 100, 100])  # 下限
    upper_red = np.array([10, 255, 255])  # 上限

    # 根据阈值提取红色像素
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    red_pixels = cv2.bitwise_and(image, image, mask=red_mask)

    return red_pixels


# 示例用法
# 加载RGB图像
image = cv2.imread('1_images/20230203181315.jpg')

# 提取红色像素
red_pixels = extract_red_pixels(image)
# 若无法提取到红色像素就认为是突变点
t = np.where(red_pixels > 0)

print(red_pixels[t])
if not red_pixels.any():
    print('找到了')
# 显示提取的红色像素
cv2.imshow('Red Pixels', red_pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()
