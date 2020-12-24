import cv2
import numpy as np


PYR_DEPTH = 5


def pyriamid_dot(pyr, c):
    for i in range(len(pyr)):
        pyr[i] = pyr[i] * c
    return pyr 


def reconstruct_pyramid(ipyr):
    ipyr = pyramid_astype(ipyr, np.uint8)
    res = ipyr[0]
    for i in ipyr[1:]:
        dstsize = (i.shape[1], i.shape[0])
        res = cv2.pyrUp(res, dstsize=dstsize) 
        res = np.add(res, i)
    return res


def gauss_pyramid(image, depth=PYR_DEPTH):
    gp = [image]
    for i in range(depth):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp


def laplacian_pyramid(image, depth=PYR_DEPTH):
    gp = gauss_pyramid(image, depth)
    gp = pyramid_astype(gp, np.int16)
    lp = [gp[depth-1]]
    for i in range(depth-1, 0, -1):
        dstsize = (gp[i-1].shape[1], gp[i-1].shape[0])
        gu = cv2.pyrUp(gp[i], dstsize=dstsize) 
        diff = np.subtract(gp[i-1], gu)
        lp.append(diff)

    return lp


def multi_pyr(w_pyr, i_pyr):
    fused_pyr = []
    for w, i in zip(w_pyr, i_pyr):
        ishape = w.shape
        ew = np.zeros((ishape[0], ishape[1], 3))
        ew[:, :, 0] = w 
        ew[:, :, 1] = w 
        ew[:, :, 2] = w
        fused = ew * i 
        fused_pyr.append(fused)
    return fused_pyr


def pyramid_astype(pyr, type):
    for i in range(len(pyr)):
        pyr[i] = np.array(pyr[i]).astype(type)
    return pyr
