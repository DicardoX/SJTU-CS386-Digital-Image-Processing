import threading
import time
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
import SIFT
import getImg
import chooseRef

exitFlag = 0

# global_list = []
threadLock = threading.Lock()


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
        # exposTime = get_exposure_time(filePath)
        if(file_extension(filePath)!='.dng'):
            currImage = cv.imread(filePath)
        elif(file_extension(filePath)=='.dng'):
            currImage = readRawImage(filePath)
        # exposTimes.append(exposTime)
        imageStack.append(currImage)
        filenames.append(file)
    exposTimes = [1.0/1000,1.0/500,1.0/250,1.0/125,1.0/60,1.0/30,1.0/15,1.0/8,1.0/4,1.0/2,1.0,2.0,3.75,7.5,15,30]
    exposTimes.reverse()
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
        print(curSaturNum)
        if curSaturNum <= saturNum:
            saturNum = curSaturNum
            refIndex = imgIndex
    # print(refIndex)
    return 9


class myThread (threading.Thread):
    def __init__(self, threadID, image, list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.index = threadID
        self.image = image
        self.list = list
    def run(self):
        # print("Processing No.", self.index, "image")
        time1 = os.times()
        sigma=1.6
        num_intervals=3
        assumed_blur=0.5
        image_border_width=5
        image = self.image.astype('float32')
        base_image = SIFT.generateBaseImage(image, sigma, assumed_blur)
        # print("Processing No.", self.index, "step1.")
        num_octaves = SIFT.computeNumberOfOctaves(base_image.shape)
        # print("Processing No.", self.index, "step2.")
        gaussian_kernels = SIFT.generateGaussianKernels(sigma, num_intervals)
        # print("Processing No.", self.index, "step3.")
        gaussian_images = SIFT.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
        # print("Processing No.", self.index, "step4.")
        dog_images = SIFT.generateDoGImages(gaussian_images)
        # print("Processing No.", self.index, "step5.")
        keypoints = SIFT.findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        # print("Processing No.", self.index, "step6.")
        keypoints = SIFT.removeDuplicateKeypoints(keypoints)
        # print("Processing No.", self.index, "step7.")
        keypoints = SIFT.convertKeypointsToInputImageSize(keypoints)
        # print("Processing No.", self.index, "step8.")
        descriptors = SIFT.generateDescriptors(keypoints, gaussian_images)
        # print("Processing No.", self.index, "finished.")

        threadLock.acquire()
        # 释放锁，开启下一个线程
        self.list.append([self.index, keypoints, descriptors])
        threadLock.release()

        time2 = os.times()

        print("Processing No.", self.index," costs ", str(time2-time1))


def SIFTAlignment(imgStack,refIndex, global_list):
    refImg = imgStack[refIndex]
    outStack = []
    idxStack = []

    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(imgStack[index])
            idxStack.append(index)
        else:
            currImg = imgStack[index]
            outImg, match_flag = SIFT.newAlignment(refImg, currImg, global_list[refIndex], global_list[index])
            if match_flag:
                outStack.append(outImg)
                idxStack.append(index)
                cv.imwrite('./mid/'+str(index)+'.jpg', outImg)
    return idxStack, outStack

def main():
    global_list = []
    time_1 = os.times()
    fileFolderPath = './test_image/Hall/'
    exposure_times,img_list,filenames = getImageStackAndExpos(fileFolderPath)


    refImgIndex= 14 # getRefImage(img_list)

    threads = []

    threadID = 0
    # for idx in range(4):
    for img in img_list:
        # img = img_list[idx]
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thread = myThread(threadID, image, global_list)
        thread.start()
        threads.append(thread)
        threadID += 1

    for t in threads:
        t.join()

    time_2 = os.times()
    print("Totoal time :", str(time_2-time_1))

    global_list.sort(key=lambda x: x[0])
    global_list = np.array(global_list[:,1:3])


    refImgIndex = chooseRef.getRefImage_br(img_list)
    cv.imwrite("ref.jpg", img_list[refImgIndex])
    res_img_list = []
    res_exp_list = []
    idx_list, res_img_list = SIFTAlignment(img_list,refImgIndex, global_list)

    for idx in idx_list:
        # res_img_list.append(image_list[idx])
        res_exp_list.append(exposure_times[idx])




    # print(global_list)

if __name__ == '__main__':
    main()