import alignment
import chooseRef
import cv2 as cv
from os import listdir
from os.path import isfile, join
import numpy as np
from Robertson import Robertson
import getImg


fileFolderPath = './test_image/Hall/'

exposTimes,image_list,filenames = getImg.getImageStackAndExpos(fileFolderPath)

# print(len(image_list))
#
images, exposTimes = alignment.process(image_list,exposTimes,'ORB')

# print(len(images))
#
# exposTimes = np.array(exposTimes,dtype=np.float32)
#
# myhdr = Robertson()
#
# myhdrPic = myhdr.process(images, exposTimes)
#
# cv.imshow('Robertson', myhdrPic)
# cv.imwrite("./res/Robertson.hdr", myhdrPic)
count = 61
for img in image_list:
    cv.imwrite("./res/hall"+str(count)+'.jpg', img)
    count += 1

