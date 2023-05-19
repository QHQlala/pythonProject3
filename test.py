import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import metrics


"""
    1.Lab是基于人对颜色的感觉来设计的，更具体地说，它是感知均匀的。
    2.在图像处理中使用较多的是 HSV 颜色空间，它比 RGB 更接近人们对彩色的感知经验。非常直观地表达颜色的色调、鲜艳程度和明暗程度，方便进行颜色的对比。
    它由三部分组成 Hue（色调、色相） Saturation（饱和度、色彩纯净度） Value（明度）
    OpenCV 中 HSV 三个分量的范围为：
        H = [0,179]
        S = [0,255]
        V = [0,255]

"""


# 直方图计算相似度
def calculate_color_similarity(image1, image2):
    # 读取图像
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # 将图像转换为HSV颜色空间
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    # 计算直方图
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # 归一化直方图
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # 计算直方图相似度
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return similarity


# 欧氏距离计算相似性
def calculate_image_similarity(image1, image2):

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # 计算颜色差异
    diff = cv2.absdiff(image1, image2)
    diff = diff.astype(float)

    dist = cv2.norm(diff)

    similarity = 1 / (1 + dist)

    return similarity


def main(image1_path, image2_path):
    # 计算颜色相似度
    similarity_hist_score = calculate_color_similarity(image1_path, image2_path)
    print('颜色相似度:{:.8f}'.format(similarity_hist_score))

    # 计算颜色相似性(每个像素点)
    similarity_ouler_score = calculate_image_similarity(image1_path, image2_path)
    print('颜色相似性:{:.8f}'.format(similarity_ouler_score))

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img = cv2.hconcat([img1, img2])
    cv2.putText(img, 'similarity of hist:{:.8f} and similarity of ouler:{:.8f}'.format(similarity_hist_score, similarity_ouler_score),
                org=(320, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 255, 0))
    # # cv2.imwrite('merge.png', img)
    # cv2.imshow('merge', img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # # 相似性大
    # image1_path = 'image.jpg'
    # image2_path = 'image1.jpg'
    # print('相似性大')
    # main(image1_path, image2_path)
    #
    # # 相似性小
    # image1_path = 'image2.jpg'
    # image2_path = 'image3.jpg'
    # print('相似性小')
    # main(image1_path, image2_path)
    hist_threshold = 0.978
    ouler_threshold = 0.0045

    for file in os.listdir('1_images'):
        img_path = '1_images' + '/' + file
        print('image.jpg' + '   ' + img_path)
        main('image.jpg', img_path)

    for file in os.listdir('2_images'):
        img_path = '2_images' + '/' + file
        print('20230203201229.jpg' + '   ' + img_path)
        main('20230203201229.jpg', img_path)


