import cv2
from pylab import *

import alignment
import getImg
import NIQE


if __name__ == '__main__':

    fileFolderPath = './test_image/Hall/'
    # exposTimes,images,filenames = getImg.getImageStackAndExpos(fileFolderPath)

    # mergeMertens = cv2.createMergeMertens()
    # exposureFusion = mergeMertens.process(images)
    # cv2.imwrite("./res/exposure-fusion.jpg", exposureFusion * 255)

    exposTimes,image_list,filenames = getImg.getImageStackAndExpos(fileFolderPath)

    print(len(image_list))

    images, exposTimes = alignment.process(image_list,exposTimes,'SIFT')

    print(len(images))

    exposTimes = np.array(exposTimes,dtype=np.float32) #convert to numpy float
    # print('exposure time:', exposTimes)

    # restore camera response
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times=exposTimes)  #获得CRF
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times=exposTimes.copy(), response=responseDebevec.copy()) #
    print("HDR NIQE score is ", NIQE.niqe(hdrDebevec))
    # Save HDR image.
    cv2.imwrite("./res/hdrDebevec.hdr", hdrDebevec)


    '''HDR photo converts to LDR'''
    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    print("ldrDrago NIQE score is ", NIQE.niqe(ldrDrago * 255))
    cv2.imwrite("./res/ldr-Drago.jpg", ldrDrago * 255)

    # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    print("ldrReinhard NIQE score is ", NIQE.niqe(ldrReinhard * 255))
    cv2.imwrite("./res/ldr-Reinhard.jpg", ldrReinhard * 255)

    # Tonemap using Mantiuk's method to obtain 24-bit color image
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    print("ldrMantiuk NIQE score is ", NIQE.niqe(ldrMantiuk * 255))
    cv2.imwrite("./res/ldr-Mantiuk.jpg", ldrMantiuk * 255)
