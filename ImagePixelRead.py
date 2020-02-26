import cv2
import numpy as np


def ImagePixelRead(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    r, c, p = img.shape
    # 索引矩阵， 检测颜色是否存在
    ind = np.zeros((256, 256, 256))
    # 颜色向量
    pixel_list = img.reshape((1, r*c, 3))
    # 存放唯一颜色的列表
    unique_pixel_list = []
    # 获取并存放唯一颜色
    for i in range(r*c):
        b, g, r = pixel_list[0, i, :]
        if ind[b, g, r] == 0:
            unique_pixel_list.append([b, g, r])
        ind[b, g, r] += 1

    # 转化为数组
    pixel_array = np.array(unique_pixel_list)

    # 计算各颜色的数量
    m, _ = pixel_array.shape
    cnt = np.zeros(m)
    for i in range(m):
        b, g, r = pixel_array[i, :]
        cnt[i] = ind[b, g, r]

    return pixel_array, cnt




