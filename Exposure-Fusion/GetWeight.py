from Utils import *
import math

import matplotlib.pyplot as plt
import matplotlib

# Used in get_well_exposedness()
SIGMA = 0.2

# Weight power
WC = 0.1
WS = 0.1
WE = 1

CSHIFT = 0
SSHIFT = 0

# constant
E = 2.7182818


def get_contrast(gray_image):
    """ Laplacian Filter"""
    dst = cv2.Laplacian(gray_image, cv2.CV_16S, ksize=3)
    abs_dst = cv2.convertScaleAbs(dst)
    
    abs_dst = np.array(abs_dst) / 255.0 + CSHIFT
    return abs_dst


def get_saturation(image):
    """ return the the standard deviation within the R,
     G and B channel, at each pixel"""
    norm_image = image / 255.0
    std = np.std(norm_image, axis=2)
    std = std + SSHIFT
    return std


def get_well_exposedness(image):
    # image = np.array(image / 255.0)
    # exposedness = -0.5*(image-0.5)**2/SIGMA**2
    # exposedness = np.power(E, exposedness)
    # exposedness = exposedness[:, :, 0] * exposedness[:, :, 1] * exposedness[:, :, 2]
    gray = rgb2gray(image) / 255
    exposedness = -0.5*(gray-0.56)**2/SIGMA**2
    exposedness = np.power(E, exposedness)
    return exposedness


def normalize_weight(weight_list):
    weight_list = np.array(weight_list) + 1e-12 # avoid division by zero
    weight_sum = np.sum(weight_list, axis=0)
    norm_weight_list = []

    " normalize "    
    for weight in weight_list:
        weight = weight / weight_sum
        norm_weight_list.append(weight)
    
    return np.array(norm_weight_list) 

def smooth_weight_list(wl):
    nwl = []
    for w in wl:
        w = cv2.GaussianBlur(w, (15, 15), 30)
        w = cv2.GaussianBlur(w, (15, 15), 30)
        nwl.append(w)
    return nwl

def get_weight(image_list):
    contrast_list = []
    saturation_list = []
    well_exposedness_list = []

    for image in image_list:
        """ get contrast """
        gray_image = rgb2gray(image)
        contrast_map = get_contrast(gray_image)
        contrast_list.append(contrast_map)
    
        """ get saturation """
        saturation_map = get_saturation(image)
        saturation_list.append(saturation_map)

        """ get well exposedness """
        exposedness_map = get_well_exposedness(image)
        well_exposedness_list.append(exposedness_map)

    """ convert to np.array """
    contrast_list = np.array(contrast_list)
    saturation_list = np.array(saturation_list)
    well_exposedness_list = np.array(well_exposedness_list)

    """ compute weight """
    weight_list = np.power(contrast_list, WC) * np.power(saturation_list, WS) + np.power(well_exposedness_list, WE)
    weight_list = smooth_weight_list(weight_list)
    normalized_weight_list = normalize_weight(weight_list)
    
    return normalized_weight_list


if __name__ == '__main__':
    image_list = get_image_stack("test_images")
    weight_list = get_weight(image_list)
    
    print("exit gracefully")
    
    