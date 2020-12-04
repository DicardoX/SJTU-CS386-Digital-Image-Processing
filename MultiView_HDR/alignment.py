import cv2
import numpy as np
# ORB image alignment
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
def imageAlignment(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("./res/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def Alignment(imgStack,refIndex):
    refImg = imgStack[refIndex]
    outStack = []
    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(refImg)
        else:
            currImg = imgStack[index]
            outImg,_ = imageAlignment(refImg,currImg)
            outStack.append(outImg)
    return outStack


# # SIFT image alignment
#
# def sift_kp(image):
#     gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d_SIFT.create()
#     kp,des = sift.detectAndCompute(image,None)
#     kp_image = cv2.drawKeypoints(gray_image,kp,None)
#     return kp_image,kp,des
#
# def get_good_match(des1,des2):
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append(m)
#     return good
#
#
# def siftImageAlignment(img1,img2):
#     _,kp1,des1 = sift_kp(img1)
#     _,kp2,des2 = sift_kp(img2)
#     goodMatch = get_good_match(des1,des2)
#     if len(goodMatch) > 4:
#         ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#         ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
#         ransacReprojThreshold = 4
#         H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
#         imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     return imgOut,H,status
