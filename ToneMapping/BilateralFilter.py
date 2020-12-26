# niqe score: 4.321003179054681 

from CRF import CRF_func, save_hdr
import cv2
import numpy as np

# 获取HDR图像
#hdr = CRF_func()
hdr = cv2.imread("taipei.hdr", flags=cv2.IMREAD_ANYDEPTH)
HSVhdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV)

def size(image):
    return image.shape[1], image.shape[0], image.shape[1] * image.shape[0]

def findMax(image):
    return max(image.reshape(image.shape[0] * image.shape[1], 1))

def findMin(image):
    return min(image.reshape(image.shape[0] * image.shape[1], 1))

def Regulation(image):
    maxItem = findMax(image)
    minItem = findMin(image)
    bias = 0.7
    x, y, N = size(image)
    for i in range(0, x, 1):
        for j in range(0, y, 1):
            image[j][i] = (np.log(image[j][i] + bias) - np.log(minItem + bias)) / (np.log(maxItem + bias) - np.log(minItem + bias)) + bias
    return image

def gamma_correction(R, G, B):
    inner = (pow(R, 2.2) + pow(1.5 * G, 2.2) + pow(0.6 * B, 2.2)) / (1 + pow(1.5, 2.2) + pow(0.6, 2.2))
    return pow(inner, 1 / 2.2)

# 获取RGB分量
BlueVector = hdr[:, :, 0]
GreenVector = hdr[:, :, 1]
RedVector = hdr[:, :, 2]
# 获取亮度分量
Lw = HSVhdr[:, :, 2]
# 获取尺寸信息
size_x, size_y, N = size(Lw)
# 色彩空间转换 + amma矫正，增加亮度
bias = 0.7
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        Lw[j][i] = (20 * RedVector[j][i] + 40 * GreenVector[j][i] + BlueVector[j][i]) / 61 + bias
# 归一化
Lw = Regulation(Lw)
# 双边滤波获得基本层
base = cv2.bilateralFilter(Lw, 5, 75, 75)       # #9为邻域直径，两个75分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
max_base = findMax(base)
min_base = findMin(base)
# 细节层
detail = Lw - base
# 计算增强因子，增强（压缩）基本层
targetContrast = np.log(5)
compressionfactor = targetContrast / (max_base - min_base)
absoluteScale = max_base * compressionfactor
# 融合
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        output_intensity = base[j][i] * compressionfactor + detail[j][i] - absoluteScale
        # 还原色彩空间
        tem = pow(10, output_intensity)
        RedVector[j][i] = RedVector[j][i] * tem / Lw[j][i]
        GreenVector[j][i] = GreenVector[j][i] * tem / Lw[j][i]
        BlueVector[j][i] = BlueVector[j][i] * tem / Lw[j][i]

        if (RedVector[j][i] <= 1):
            RedVector[j][i] *= 255
        else:
            RedVector[j][i] = 255
        if (GreenVector[j][i] <= 1):
            GreenVector[j][i] *= 255
        else:
            GreenVector[j][i] = 255
        if (BlueVector[j][i] <= 1):
            BlueVector[j][i] *= 255
        else:
            BlueVector[j][i] = 255

# Gamma矫正
for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        Lw[j][i] = gamma_correction(RedVector[j][i], GreenVector[j][i], BlueVector[j][i])
# hdr[:, :, 0] = BlueVector
# hdr[:, :, 1] = GreenVector
# hdr[:, :, 2] = RedVector
HSVhdr[:, :, 2] = Lw
hdr = cv2.cvtColor(HSVhdr, cv2.COLOR_HSV2BGR)
# 中值滤波以去除椒盐噪声
hdr = cv2.medianBlur(hdr, 3)
# 锐化
sharpen_op = np.array([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 2, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]], dtype=np.float32)
hdr = cv2.filter2D(hdr, cv2.CV_32F, sharpen_op)
# 保存hdr
save_hdr(hdr, "hdr.jpg")




