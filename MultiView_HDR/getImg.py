from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from PIL.ExifTags import TAGS
import PIL.ExifTags
import cv2
import os.path
import exifread
import rawpy

# RAW to BGR
def readRawImage(filename):
    with rawpy.imread(filename) as raw:
        rgb = raw.postprocess()
    # Changing to BGR
    rgb[:, :, 0], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
    return rgb

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