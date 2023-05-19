import os

import cv2


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


hist_threshold = 0.9775


# 以第一张图片为基准, 判断与其他图片的直方图相似度
def main(baseline, images_path):
    for file in os.listdir(images_path):
        img_path = images_path + '/' + file
        similarity = calculate_color_similarity(baseline, img_path)
        if similarity < hist_threshold:
            print('突变点图片编号: {}'.format(img_path))
            break

if __name__ == '__main__':
    baseline = 'image.jpg'
    main(baseline, '1_images')

    baseline = '20230203201229.jpg'
    main(baseline, '2_images')


