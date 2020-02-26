from ImagePixelRead import *
import matplotlib.pyplot as plt
from min_result import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

filename = 'images/parrots.png'

unique_pixel, color_cnt = ImagePixelRead(filename)

np.savetxt('unique_pixel.txt', unique_pixel)

k = 64
# 构造聚类器
estimator = MiniBatchKMeans(n_clusters=k, init_size=3*k, batch_size=2000)
# 聚类
estimator.fit(unique_pixel)
# 获取聚类标签
label_pred = estimator.labels_
np.savetxt('label_pre.txt', label_pred)
# 获取聚类中心
centroids = estimator.cluster_centers_
# 获取聚类准则的总和
inertia = estimator.inertia_
ax = plt.figure().add_subplot(111, projection='3d')
for i in range(k):
    ind = label_pred == i
    ax.scatter(unique_pixel[ind, 0], unique_pixel[ind, 1], unique_pixel[ind, 2])
plt.show()
# 获取优化后的质心
min_center = min_result(unique_pixel, label_pred, k, color_cnt)

# 重构图片
img = cv2.imread(filename, cv2.IMREAD_COLOR)
r, c, p = img.shape
mapping_color = np.zeros((256, 256, 256))
m = unique_pixel.shape[0]

for i in range(m):
    b, g, r_ = unique_pixel[i, :]
    mapping_color[int(b), int(g), int(r_)] = label_pred[i]

for row in range(r):
    for col in range(c):
        b, g, r_ = img[row, col, :]
        ind = int(mapping_color[b, g, r_])
        img[row, col, :] = min_center[ind, :]

cv2.imwrite('6_64.png', img)

