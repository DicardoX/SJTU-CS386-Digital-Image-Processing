import cv2
from pylab import *

import alignment
import getImg
import chooseRef

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
    exposTimes,images,filenames = getImg.getImageStackAndExpos(fileFolderPath)

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
    exposTimes,images,filenames = getImg.getImageStackAndExpos(fileFolderPath)
    print(exposTimes)
    print(filenames)
    refImgIndex= chooseRef.getRefImage(images)
    images = alignment.Alignment(images,refImgIndex)
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
