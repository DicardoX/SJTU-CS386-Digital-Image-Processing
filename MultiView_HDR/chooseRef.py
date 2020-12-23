import cv2
import numpy as np
import quality

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
    # print(refIndex)
    return  refIndex

def getRefImage_br(imgStack):
    bright_list = []
    # for img in imgStack:
    #     k = quality.bright(img)
    #     bright_list.append(k)
    #
    # refIndex = bright_list.index(min(bright_list))

    for img in imgStack:
        k = quality.getImageVar(img)
        bright_list.append(k)

    refIndex = bright_list.index(max(bright_list))
    print(refIndex)
    return  refIndex
