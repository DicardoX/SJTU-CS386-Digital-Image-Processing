from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from PIL.ExifTags import TAGS
import PIL.ExifTags
import cv2
import os.path
import rawpy
import numpy as np

def show_image(image, name="Display"):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, image)
    cv2.waitKey(0)

def list_files(FilePath):
    onlyfiles = [f for f in listdir(FilePath) if isfile(join(FilePath, f))]
    return onlyfiles

def read_raw_image(filename):
    with rawpy.imread(filename) as raw:
        rgb = raw.postprocess()
    # Changing to BGR
    rgb[:, :, 0], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
    return rgb

# retrieve file extension
def file_extension(path):
    return os.path.splitext(path)[1]

def get_image_stack(folderPath):
    files = list_files(folderPath)
    imageStack = []
    for file in files:
        filePath = join(folderPath,file)
        if(file_extension(filePath)=='.jpg'):
            currImage = cv2.imread(filePath)
        elif(file_extension(filePath)=='.dng'):
            currImage = read_raw_image(filePath)
        imageStack.append(currImage)
    return np.array(imageStack)

def rgb2gray(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    return gray_image

def rgb2yuv(rgb_image):
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)
    return yuv_image

def yuv2rgb(yuv_image):
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    return rgb_image


if __name__ == '__main__':
    image_list = get_image_stack("test_images")
    image = image_list[0]
    yuv = rgb2yuv(image)
    rgb = yuv2rgb(image)
    pass
