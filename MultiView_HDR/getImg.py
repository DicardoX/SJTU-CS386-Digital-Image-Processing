from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from PIL.ExifTags import TAGS
import PIL.ExifTags
import cv2

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
