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
    return exposure_time

#获取图像曝光时间序列和图像
def getImageStackAndExpos(folderPath):
    files = ListFiles(folderPath)
    exposTimes = []
    imageStack = []
    filenames = []
    for file in files:
        filePath = join(folderPath,file)
        exposTime = get_exposure_time(filePath)
        currImage = cv2.imread(filePath)
        exposTimes.append(exposTime)
        imageStack.append(currImage)
        filenames.append(file)
    #根据曝光时间长短，对图像序列和曝光时间序列重新排序
    index = sorted(range(len(exposTimes)), key=lambda k: exposTimes[k])
    exposTimes = [exposTimes[i] for i in index]
    imageStack = [imageStack[i] for i in index]
    filenames = [filenames[i] for i in index]
    return exposTimes,imageStack,filenames

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
        # print(curSaturNum)
        if curSaturNum <= saturNum:
            saturNum = curSaturNum
            refIndex = imgIndex

    return  refIndex

# ORB image alignment
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
def imageAlignment(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("./res/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

# MTB image alignment
def MTBAlignment(im1, im2):
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(im1, im2)


# # SIFT image alignment
#
# def sift_kp(image):
#     gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d_SIFT.create()
#     kp,des = sift.detectAndCompute(image,None)
#     kp_image = cv2.drawKeypoints(gray_image,kp,None)
#     return kp_image,kp,des
#
# def get_good_match(des1,des2):
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append(m)
#     return good
#
# '''cv2.ORB_create()'''
#
# def siftImageAlignment(img1,img2):
#     _,kp1,des1 = sift_kp(img1)
#     _,kp2,des2 = sift_kp(img2)
#     goodMatch = get_good_match(des1,des2)
#     if len(goodMatch) > 4:
#         ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#         ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#         ransacReprojThreshold = 4
#         H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
#         imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     return imgOut,H,status

def Alignment(imgStack,refIndex):
    refImg = imgStack[refIndex]
    outStack = []
    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(refImg)
        else:
            currImg = imgStack[index]
            outImg,_ = imageAlignment(refImg,currImg)
            outStack.append(outImg)
    return outStack

if __name__ == '__main__':

    # filename = "./test_image/" + input("Please input the filename of the picture: ")
    # ft1=filetype.guess(filename)
    # if ft1 is None:
    #     print('无法判断该文件类型')
    # print('文件扩展名为：{}'.format(ft1.extension))
    # print('文件类型为：{}'.format(ft1.mime))
    # fileExt = format(ft1.extension)
    # JPGexifread(filename)
    # if(fileExt == 'jpg'):
    #     JPGexifread(filename)
    # if(fileExt == 'tif'):
    fileFolderPath = './test_image/myTest/'
    exposTimes,images,filenames = getImageStackAndExpos(fileFolderPath)

    # # MTB
    # # Align input images
    # alignMTB = cv2.createAlignMTB()
    # alignMTB.process(images, images)
    # for i in range(0, len(images)):
    #     cv2.imwrite('./mid/'+filenames[i], images[i])
    #
    # mergeMertens = cv2.createMergeMertens()
    # exposureFusion = mergeMertens.process(images)
    # cv2.imwrite("./res/exposure-fusion.jpg", exposureFusion * 255)

    # ORB
    exposTimes,images,filenames = getImageStackAndExpos(fileFolderPath)
    print(exposTimes)
    print(filenames)
    refImgIndex= getRefImage(images)
    images = Alignment(images,refImgIndex)
    for i in range(0, len(images)):
        cv2.imwrite('./mid/'+filenames[i], images[i])

    exposTimes = np.array(exposTimes,dtype=np.float32) #需要转化为numpy浮点数组
    # print('exposure time:', exposTimes)

    # restore camera response
    calibrateDebevec = cv2.createCalibrateDebevec()
    # calibrateDebevec = cv2.createCalibrateDebevec(samples=150,random=True)
    ###采样点数1000个，采样方式为随机，一般而言，采用点数越多，采样方式越随机，最后的CRF曲线会越加平滑
    responseDebevec = calibrateDebevec.process(images, exposTimes)  #获得CRF
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, exposTimes.copy(), responseDebevec.copy()) #
    # Save HDR image.
    cv2.imwrite("./res/hdrDebevec.hdr", hdrDebevec)

    # # Robertson method
    calibrateRobertson = cv2.createCalibrateRobertson()
    responseRobertson = calibrateRobertson.process(images, exposTimes)
    mergeRobertson = cv2.createMergeRobertson()
    hdrRobertson = mergeRobertson.process(images, exposTimes, responseRobertson)
    cv2.imwrite("./res/hdrRobertson.hdr", hdrRobertson)

    '''HDR photo converts to LDR'''

    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("./res/ldr-Drago.jpg", np.clip(ldrDrago * 255, 0, 255).astype('uint8'))

    # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite("./res/ldr-Reinhard.jpg", np.clip(ldrReinhard * 255, 0, 255).astype('uint8'))

    # Tonemap using Mantiuk's method to obtain 24-bit color image
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite("./res/ldr-Mantiuk.jpg", np.clip(ldrMantiuk * 255, 0, 255).astype('uint8'))
