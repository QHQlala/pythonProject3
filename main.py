import cv2
import numpy as np

# 加载图像
image = cv2.imread('image.jpg')

# 将图像转换为Lab颜色空间
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# 分割图像为主要区域和背景
# 这里使用简单的阈值方法进行分割
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# 获取主要区域的颜色变化
# 计算主要区域的颜色直方图
hist = cv2.calcHist([image_lab], [1, 2], threshold, [256, 256], [0, 256, 0, 256])

# 比较颜色直方图之间的差异
# 这里使用相关性作为相似度度量
correlation = cv2.compareHist(hist, hist, cv2.HISTCMP_CORREL)

# 标记颜色变化区域
# 根据相关性阈值确定颜色变化的区域
threshold_correlation = 0.9
color_change_mask = np.where(correlation < threshold_correlation, 255, 0).astype(np.uint8)

# 显示结果
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.imshow('Color Change Mask', color_change_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
