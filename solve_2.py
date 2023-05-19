import os

import cv2


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

ouler_threshold = 0.0000388


# 以第一张图片为基准, 判断与其他图片的直方图相似度
def main(baseline, images_path):
    for file in os.listdir(images_path):
        img_path = images_path + '/' + file
        similarity = calculate_image_similarity(baseline, img_path)
        if similarity < ouler_threshold:
            print('突变点图片编号: {}'.format(img_path))
            break

if __name__ == '__main__':
    baseline = 'image.jpg'
    main(baseline, '1_images')

    baseline = '20230203201229.jpg'
    main(baseline, '2_images')