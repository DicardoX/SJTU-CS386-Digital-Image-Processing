import cv2 as cv
import numpy as np
import os
import chooseRef


'''ORB image alignment'''
MAX_MATCHES = 50000
GOOD_MATCH_PERCENT = 0.05

def ORBimageAlignment(im2, im1):

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
    print(len(matches))
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
    height, width, channels = im2.shape
    im1Reg = cv.warpPerspective(im1, h, (width, height))

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
def ECCimageAlignment(im1, im2):
    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Find size of image1
    im1_size = im1.shape

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
        im2_aligned = cv.warpPerspective(im2, warp_matrix, (im1_size[1], im1_size[0]),
                                         flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv.warpAffine(im2, warp_matrix, (im1_size[1], im1_size[0]),
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


def process(image_list, exposure_times, mode):
    if(mode=='MTB'):
        # MTB
        alignMTB = cv.createAlignMTB()
        alignMTB.process(image_list, image_list)
        return image_list, exposure_times
    elif(mode=='ORB'):
        refImgIndex= chooseRef.getRefImage(image_list)
        return ORBAlignment(image_list,refImgIndex,exposure_times)
    elif(mode=='grad'):
        # todo
        return image_list, exposure_times
    elif(mode=='ECC'):
        refImgIndex= chooseRef.getRefImage(image_list)
        return ECCAlignment(image_list,refImgIndex), exposure_times
    else:
        return image_list, exposure_times
