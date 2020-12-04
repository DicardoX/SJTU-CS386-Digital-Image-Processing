import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from pylab import *
import skimage
from skimage import metrics
from PIL import Image
from PIL.ExifTags import TAGS
import PIL.ExifTags
import exifread
import filetype
# from libtiff import TIFF
from os import listdir
from os.path import isfile, isdir, join


'''PART ONE: Tiff photo process'''
# #读取文件夹下文件
def ListFiles(FilePath):
    onlyfiles = [f for f in listdir(FilePath) if isfile(join(FilePath, f))]
    return onlyfiles

#获得图像文件属性
def get_exif(fn):
    img = Image.open(fn)
    exif = {PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
            }
    return exif

#获得图像曝光时间
def get_exposure_time(fn):
    exif = get_exif(fn)
    exposure_time = exif.get('ExposureTime')
    return exposure_time[0]/exposure_time[1]

#获取图像曝光时间序列和图像
def getImageStackAndExpos(folderPath):
    files = ListFiles(folderPath)
    exposTimes = []
    imageStack = []
    for file in files:
        filePath = join(folderPath,file)
        exposTime = get_exposure_time(filePath)
        currImage = cv2.imread(filePath)
        exposTimes.append(exposTime)
        imageStack.append(currImage)
    #根据曝光时间长短，对图像序列和曝光时间序列重新排序
    index = sorted(range(len(exposTimes)), key=lambda k: exposTimes[k])
    exposTimes = [exposTimes[i] for i in index]
    imageStack = [imageStack[i] for i in index]
    return exposTimes,imageStack

# preprocess: choose the standard photo
def getSaturNum(img):
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    underExpos=np.count_nonzero(gray_image==0)
    overExpos = np.count_nonzero(gray_image==255)
    return underExpos + overExpos


def getRefImage(imgStack):
    saturNum = imgStack[0].shape[0]*imgStack[0].shape[1]
    for imgIndex in np.arange(len(imgStack)):
        curImg = imgStack[imgIndex]
        curSaturNum = getSaturNum(curImg)
        print(curSaturNum)
        if curSaturNum <= saturNum:
            saturNum = curSaturNum
            refIndex = imgIndex

    return  refIndex

# SIFT image alignment
def siftAlignment(imgStack,refIndex):
    refImg = imgStack[refIndex]
    outStack = []
    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(refImg)
        else:
            currImg = imgStack[index]
            outImg,_,_ = siftImageAlignment(refImg,currImg)
            outStack.append(outImg)
    return outStack

'''PART TWO: Jpg photo process'''
def readImagesAndTimes(images):
    # List of exposure times
    times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
    # List of image filenames
    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
    # images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)
    return images, times

def JPGexifread(filename):
    for (k,v) in Image.open(filename)._getexif().items():
        if(TAGS.get(k)=='ExposureTime'):
            print('%s = %s' % (TAGS.get(k),v))

    f = open(filename, 'rb')
    tags = exifread.process_file(f)
    for tag in tags.keys():
        if(tag=='EXIF ExposureTime'):
            print(tag, tags[tag])


if __name__ == '__main__':

    filename = "./test_image/" + input("Please input the filename of the picture: ")
    ft1=filetype.guess(filename)
    if ft1 is None:
        print('无法判断该文件类型')
    print('文件扩展名为：{}'.format(ft1.extension))
    print('文件类型为：{}'.format(ft1.mime))
    fileExt = format(ft1.extension)
    # JPGexifread(filename)
    # if(fileExt == 'jpg'):
    #     JPGexifread(filename)
    if(fileExt == 'tif'):
        exposTimes,images = getImageStackAndExpos('stack_alignment')
        refImgIndex= getRefImage(images)
        images = siftAlignment(images,refImgIndex)

        exposTimes = np.array(exposTimes,dtype=np.float32) #需要转化为numpy浮点数组
        calibrateDebevec = cv2.createCalibrateDebevec(samples=120,random=True)
        ###采样点数120个，采样方式为随机，一般而言，采用点数越多，采样方式越随机，最后的CRF曲线会越加平滑
        responseDebevec = calibrateDebevec.process(images, exposTimes)  #获得CRF
        mergeDebevec = cv2.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(images, exposTimes, responseDebevec) #
        # Save HDR image.
        cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
        # Tonemap using Drago's method to obtain 24-bit color image
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        ldrDrago = tonemapDrago.process(hdrDebevec)
        ldrDrago = 3 * ldrDrago
        cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)

# images = []
    # readImagesAndTimes()
    #
    # # Align input images
    # alignMTB = cv2.createAlignMTB()
    # alignMTB.process(images, images)
    #
    # # Obtain Camera Response Function (CRF)
    # calibrateDebevec = cv2.createCalibrateDebevec()
    # responseDebevec = calibrateDebevec.process(images, times)
    #
    # # Merge images into an HDR linear image
    # mergeDebevec = cv2.createMergeDebevec()
    # hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    # # Save HDR image.
    # cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
    #
    # # Tonemap using Drago's method to obtain 24-bit color image
    # tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    # ldrDrago = tonemapDrago.process(hdrDebevec)
    # ldrDrago = 3 * ldrDrago
    # cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)