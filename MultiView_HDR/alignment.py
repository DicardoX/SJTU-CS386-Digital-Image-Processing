import cv2 as cv
import numpy as np
import os
import chooseRef
import multiThreads
import SIFT


'''ORB image alignment'''
MAX_MATCHES = 50000
GOOD_MATCH_PERCENT = 0.05

def ORBimageAlignment(ori_img2, ori_img1):

    im1 = equalize(ori_img1)
    im2 = equalize(ori_img2)

    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # print(len(matches))
    if(len(matches)<1000):
        return im1,None,False
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv.imwrite("./mid/"+str(os.times())+"matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, channels = ori_img2.shape
    im1Reg = cv.warpPerspective(ori_img1, h, (width, height))

    return im1Reg,h,True


def ORBAlignment(imgStack,refIndex,exposure_times):
    refImg = imgStack[refIndex]
    outStack = []
    exposureStack = []
    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(refImg)
            exposureStack.append(exposure_times[index])
        else:
            currImg = imgStack[index]
            outImg,_,flag = ORBimageAlignment(refImg,currImg)
            if(flag):
                outStack.append(outImg)
                exposureStack.append(exposure_times[index])
                cv.imwrite('./mid/'+str(index)+'.jpg', outImg)
    return outStack, exposureStack

'''ECC Alignment'''
def ECCimageAlignment(ori_img1, ori_img2):
    im1 = equalize(ori_img1)
    im2 = equalize(ori_img2)

    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Find size of image1
    im1_size = ori_img1.shape

    # Define the motion model
    warp_mode = cv.MOTION_TRANSLATION
    # warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Enhanced Correlation Coefficient (ECC)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 5)

    if warp_mode == cv.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv.warpPerspective(ori_img2, warp_matrix, (im1_size[1], im1_size[0]),
                                         flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv.warpAffine(ori_img2, warp_matrix, (im1_size[1], im1_size[0]),
                                    flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

    return im2_aligned, 0

def ECCAlignment(imgStack,refIndex):
    refImg = imgStack[refIndex]
    outStack = []
    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(refImg)
        else:
            currImg = imgStack[index]
            outImg,_ = ECCimageAlignment(refImg,currImg)
            outStack.append(outImg)
            cv.imwrite('./mid/'+str(index)+'.jpg', outImg)
    return outStack


'''AKAZE'''
def equalize(image):
    chans = cv.split(image)
    colors = ("b", "g", "r")
    for (chan, color) in zip(chans, colors):
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        # plt.plot(hist, color=color)
        # plt.xlim([0, 256])

    equalizeImg = np.zeros(image.shape, image.dtype)
    equalizeImg[:, :, 0] = cv.equalizeHist(image[:, :, 0])
    equalizeImg[:, :, 1] = cv.equalizeHist(image[:, :, 1])
    equalizeImg[:, :, 2] = cv.equalizeHist(image[:, :, 2])

    return equalizeImg


def AKAZEimageAlignment(ori_img1, ori_img2):
    # ori_img1 = cv.imread('./test_image/myTest/StLouisArchMultExpEV-1.82.jpg')  # referenceImage
    # ori_img2 = cv.imread('./test_image/myTest/StLouisArchMultExpEV+4.09.jpg')  # sensedImage

    img1 = equalize(ori_img1)
    img2 = equalize(ori_img2)

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Initiate AKAZE detector
    akaze = cv.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    if len(good_matches)<7:
        return ori_img1, False
    else:
        # # Draw matches
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite('matches.jpg', img3)

        # Select good matched keypoints
        ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Compute homography
        H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC,5.0)

        if H is None:
            return ori_img1, False
        else:
            # Warp image
            # print("match!")
            warped_image = cv.warpPerspective(ori_img1, H, (ori_img2.shape[1], ori_img2.shape[0]))

            # cv.imwrite('warped.jpg', warped_image)
            return warped_image, True

def AKAZEAlignment(imgStack,refIndex):
    refImg = imgStack[refIndex]
    outStack = []
    idxStack = []

    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(imgStack[index])
            idxStack.append(index)
        else:
            currImg = imgStack[index]
            outImg, match_flag = AKAZEimageAlignment(currImg, refImg)
            if match_flag:
                outStack.append(outImg)
                idxStack.append(index)
                cv.imwrite('./mid/'+str(index)+'.jpg', outImg)
    return idxStack, outStack


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


def SIFTProcess(img_list, exposure_times):
    global_list = []
    time_1 = os.times()
    threads = []

    threadID = 0
    # for idx in range(4):
    for img in img_list:
        # img = img_list[idx]
        image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thread = multiThreads.myThread(threadID, image, global_list)
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

    return res_img_list, res_exp_list

def process(image_list, exposure_times, mode):
    if(mode=='MTB'):
        # MTB
        alignMTB = cv.createAlignMTB()
        alignMTB.process(image_list, image_list)
        return image_list, exposure_times
    elif(mode=='ORB'):
        refImgIndex = chooseRef.getRefImage_br(image_list)
        cv.imwrite("ref.jpg", image_list[refImgIndex])
        return ORBAlignment(image_list,refImgIndex,exposure_times)
    elif(mode=='grad'):
        # todo
        return image_list, exposure_times
    elif(mode=='ECC'):
        refImgIndex= chooseRef.getRefImage_br(image_list)
        cv.imwrite("ref.jpg", image_list[refImgIndex])
        return ECCAlignment(image_list,refImgIndex), exposure_times
    elif(mode=='AKAZE'):
        # refImgIndex = chooseRef.getRefImage(image_list)
        refImgIndex = chooseRef.getRefImage_br(image_list)
        cv.imwrite("ref.jpg", image_list[refImgIndex])
        res_img_list = []
        res_exp_list = []
        idx_list, res_img_list = AKAZEAlignment(image_list,refImgIndex)
        for idx in idx_list:
            # res_img_list.append(image_list[idx])
            res_exp_list.append(exposure_times[idx])
        return res_img_list, res_exp_list
    elif(mode=='SIFT'):
        return SIFTProcess(image_list, exposure_times)
    else:
        return image_list, exposure_times
