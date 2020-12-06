from Utils import *
import math

import matplotlib.pyplot as plt
import matplotlib

# Used in get_well_exposedness()
SIGMA = 0.2

# Weight power
WC = 1
WS = 1
WE = 0.2

CSHIFT = 0.3
SSHIFT = 0.1

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

    def gauss_func(x):
        t = -(x-0.5)**2/SIGMA**2
        return math.exp(t)

    # if image is rgb, first convert it to gray style
    if len(image.shape)==3 and image.shape[2]==3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image

    gray_image = gray_image / 255.0
    # for loop is too slow
    # for i in range(gray_image.shape[0]):
    #     for j in range(gray_image.shape[1]):
    #         pixel = gray_image[i][j]
    #         pixel = gauss_func(pixel)
    #         gray_image[i][j] = pixel
    exposed = np.array(gray_image)
    exposed = -(exposed-0.5)**2/SIGMA**2
    exposed = np.power(E, exposed)
    return exposed


def normalize_weight(weight_list):
    weight_sum = np.sum(weight_list, axis=0)
    weight_sum += 1e-10 # avoid divide by zero
    norm_weight_list = []

    # plt.hist(weight_sum.flatten(), facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.show()

    " normalize "    
    for weight in weight_list:
        weight = weight / weight_sum
        norm_weight_list.append(weight)
    
    #show_image(weight_list[0], "before normalize, 0")
    #show_image(norm_weight_list[0], "after normalize, 0")
    # show_image(weight_list[1], "after normalize, 1")
    #show_image(weight_list[1], "after normalize, 1")
    # show_image(weight_list[2], "after normalize, 2")
    #show_image(weight_list[2], "after normalize, 2")
    
    return np.array(norm_weight_list) 


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
        exposedness_map = get_well_exposedness(gray_image)
        well_exposedness_list.append(exposedness_map)

    """ convert to np.array """
    contrast_list = np.array(contrast_list)
    saturation_list = np.array(saturation_list)
    well_exposedness_list = np.array(well_exposedness_list)

    """ compute weight """
    weight_list = np.power(contrast_list, WC) * np.power(saturation_list, WS) * np.power(well_exposedness_list, WE)
    normalized_weight_list = normalize_weight(weight_list)
    
    return normalized_weight_list


if __name__ == '__main__':
    image_list = get_image_stack("test_images")
    weight_list = get_weight(image_list)
    
    print("exit gracefully")
    
    
