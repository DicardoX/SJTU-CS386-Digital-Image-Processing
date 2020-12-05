from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from PIL.ExifTags import TAGS
import PIL.ExifTags
import cv2 as cv
import numpy as np
import os.path
import exifread
import rawpy

# RAW to BGR
def readRawImage(filename):
    raw = rawpy.imread(filename)
    img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8)

    rgb = np.float32(img / 65535.0*255.0)
    rgb = np.asarray(rgb,np.uint8)

    file_name = os.path.splitext(filename)[0]
    file_name = os.path.split(file_name)[1]
    # print(file_name)
    cv.imwrite('./mid/rgb_'+file_name+'.jpg', img)
    cv.imwrite('./mid/'+file_name+'.jpg', img)
    # with rawpy.imread(filename) as raw:
    #     rgb = raw.postprocess()
    # # Changing to BGR
    # rgb[:, :, 0], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
    # cv.imwrite('./mid/changed_'+filename, rgb)
    return img

# retrieve file extension
def file_extension(path):
    return os.path.splitext(path)[1]

# #读取文件夹下文件
def ListFiles(FilePath):
    onlyfiles = [f for f in listdir(FilePath) if isfile(join(FilePath, f))]
    return onlyfiles

#获得图像曝光时间
def get_exposure_time(fn):
    # print(file_extension(fn))
    if(file_extension(fn)=='.jpg'):
        img = Image.open(fn)
        exif = {PIL.ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in PIL.ExifTags.TAGS
                }
        return exif.get('ExposureTime')
    else:
        raw_file = open(fn, 'rb')
        exif_file = exifread.process_file(raw_file, details=False, strict=True)
        # print(exif_file)

        if('EXIF ExposureTime' in exif_file.keys()):
            exposure_str = exif_file['EXIF ExposureTime'].printable
        else:
            exposure_str = exif_file['Image ExposureTime'].printable

        if '/' in exposure_str:
            fenmu = float(exposure_str.split('/')[0])
            fenzi = float(exposure_str.split('/')[-1])
            exposure = fenmu / fenzi
        else:
            exposure = float(exposure_str)
        return exposure

#获取图像曝光时间序列和图像
def getImageStackAndExpos(folderPath):
    files = ListFiles(folderPath)
    exposTimes = []
    imageStack = []
    filenames = []
    for file in files:
        filePath = join(folderPath,file)
        exposTime = get_exposure_time(filePath)
        if(file_extension(filePath)=='.jpg'):
            currImage = cv.imread(filePath)
        elif(file_extension(filePath)=='.dng'):
            currImage = readRawImage(filePath)
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
    gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
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
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv.imwrite("./res/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, h, (width, height))

    return im1Reg, h

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
    fileFolderPath = './input/'
    exposure_times,img_list,filenames = getImageStackAndExpos(fileFolderPath)

    # Align input images
    alignMTB = cv.createAlignMTB()
    alignMTB.process(img_list, img_list)

    # refImgIndex= getRefImage(img_list)
    # img_list = Alignment(img_list,refImgIndex)

    exposure_times = np.array(exposure_times,dtype=np.float32) #需要转化为numpy浮点数组
    # Merge exposures to HDR image
    # Estimate camera response function (CRF)
    merge_debevec = cv.createMergeDebevec()
    cal_debevec = cv.createCalibrateDebevec()
    crf_debevec = cal_debevec.process(img_list, times=exposure_times)
    hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy(), response=crf_debevec.copy())
    cv.imwrite("./res/hdrDebevec.hdr", hdr_debevec)

    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdr_debevec)
    ldrDrago = 3 * ldrDrago
    cv.imwrite("./res/ldr-Drago.jpg", ldrDrago * 255)


    # Exposure fusion using Mertens
    merge_mertens = cv.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)



    # Convert datatype to 8-bit and save
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
    cv.imwrite("./res/fusion_mertens.jpg", res_mertens_8bit)