# niqe score: 4.92555694064178

from CRF import CRF_func, save_hdr
import cv2
import math

# 获取HDR图像
#hdr = CRF_func()
hdr = cv2.imread("taipei.hdr", flags=cv2.IMREAD_ANYDEPTH)
#hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV)

def size(image):
    return image.shape[1], image.shape[0], image.shape[1] * image.shape[0]

def average(image):
    sum = 0
    for i in range(0, size_x, 1):
        for j in range(0, size_y, 1):
            sum += image[j][i]
    sum /= N
    avgLw = sum
    return avgLw

def toneMapping(image, avg, alpha, Lwhite):
    Lw = image * (alpha / avg)
    for i in range(0, size_x, 1):
        for j in range(0, size_y, 1):
            Lw[j][i] = (Lw[j][i] * (1 + Lw[j][i] / pow(Lwhite, 2))) / (1 + Lw[j][i])
    return Lw

# 获取亮度分量
BlueVector = hdr[:, :, 0]
GreenVector = hdr[:, :, 1]
RedVector = hdr[:, :, 2]
# 获取尺寸信息
size_x, size_y, N = size(BlueVector)

# 求平均亮度：log-average
avgBlue = average(BlueVector)
avgGreen = average(GreenVector)
avgRed = average(RedVector)

# # 求原图片亮度最大值
# Hmax = 0
# for i in range(0, size_x, 1):
#     for j in range(0, size_y, 1):
#         if (Hmax < Lw[j][i]):
#             Hmax = Lw[j][i]
#
# # 求原图片亮度最小值
# Hmin = 1000
# for i in range(0, size_x, 1):
#     for j in range(0, size_y, 1):
#         if (Hmin > Lw[j][i]):
#             Hmin = Lw[j][i]

# for i in range(0, size_x, 1):
#     for j in range(0, size_y, 1):
#         Lw[j][i] = ((Lw[j][i] - Hmin) / (Hmax - Hmin)) * 255

# 色调映射
alpha = 0.36
Lwhite = 240
# Lw = Lw * (alpha / avgLw)
hdr[:, :, 0] = toneMapping(BlueVector, avgBlue, alpha, Lwhite)
hdr[:, :, 1] = toneMapping(GreenVector, avgGreen, alpha, Lwhite)
hdr[:, :, 2] = toneMapping(RedVector, avgRed, alpha, Lwhite)

#hdr[:, :, 2] = Lw
#hdr = cv2.cvtColor(hdr, cv2.COLOR_HSV2BGR)

# 保存hdr
save_hdr(hdr, "hdr.jpg")




