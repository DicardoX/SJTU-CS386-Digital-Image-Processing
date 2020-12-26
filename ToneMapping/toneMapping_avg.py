# niqe score: 4.62616037411216

from CRF import CRF_func, save_hdr
import cv2
import math

# 获取HDR图像
#hdr = CRF_func()
hdr = cv2.imread("taipei.hdr", flags=cv2.IMREAD_ANYDEPTH)
hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV)

# 获取亮度分量
Lw = hdr[:, :, 2]
# 获取总像素数
size_x = Lw.shape[1]
size_y = Lw.shape[0]
N = Lw.shape[0] * Lw.shape[1]

# 求平均亮度：log-average
bias = 0.1
sum = 0
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        #sum += math.log(bias + Lw[j][i])
        sum += Lw[j][i]
sum /= N
#avgLw = math.exp(sum)
avgLw = sum

# # 求原图片亮度最大值
Hmax = 0
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        if (Hmax < Lw[j][i]):
            Hmax = Lw[j][i]

# 求原图片亮度最小值
Hmin = 1000
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        if (Hmin > Lw[j][i]):
            Hmin = Lw[j][i]

# for i in range(0, size_x, 1):
#     for j in range(0, size_y, 1):
#         Lw[j][i] = ((Lw[j][i] - Hmin) / (Hmax - Hmin)) * 255

# 色调映射
alpha = 0.36
Lwhite = 10
Lw = Lw * (alpha / avgLw)
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        Lw[j][i] = (Lw[j][i] * (1 + Lw[j][i] / pow(Lwhite, 2))) / (1 + Lw[j][i])

hdr[:, :, 2] = Lw
hdr = cv2.cvtColor(hdr, cv2.COLOR_HSV2BGR)

# 保存hdr
save_hdr(hdr, "hdr.jpg")




