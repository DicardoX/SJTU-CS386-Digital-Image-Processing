import math
from os import listdir
from os.path import isfile, join

import cv2 as cv
import numpy as np


#利用拉普拉斯
def getImageVar(image):
    img2gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageVar = cv.Laplacian(img2gray, cv.CV_64F).var()
    print(imageVar)
    return imageVar

def bright(image):
    # 把图片转换为单通道的灰度图
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    ma = 0
    #np.full 构造一个数组，用指定值填充其元素
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i-128-da) * hist[i])
    m = abs(ma / size)
    # 亮度系数
    k = abs(da) / m

    # print(k)
    if k[0] > 1:
        # 过亮
        if da > 0:
            print("过亮")
        else:
            print("过暗")
    else:
        print("亮度正常")
    return k[0]

def color_cast(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(image)
    h,w,_ = image.shape
    da = a_channel.sum()/(h*w)-128
    db = b_channel.sum()/(h*w)-128
    histA = [0]*256
    histB = [0]*256
    for i in range(h):
        for j in range(w):
            ta = a_channel[i][j]
            tb = b_channel[i][j]
            histA[ta] += 1
            histB[tb] += 1
    msqA = 0
    msqB = 0
    for y in range(256):
        msqA += float(abs(y-128-da))*histA[y]/(w*h)
        msqB += float(abs(y - 128 - db)) * histB[y] / (w * h)

    result = math.sqrt(da*da+db*db)/math.sqrt(msqA*msqA+msqB*msqB)

    print("d/m = %s"%result)
    return  result


# folderPath = './test_image/myTest/'
# files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
# print(len(files))
# for file in files:
#     filePath = join(folderPath,file)
#     print(filePath)
#     currImage = cv.imread(filePath)
#
#     getImageVar(currImage)
#     bright(currImage)
#     color_cast(currImage)
#     print('\n')


