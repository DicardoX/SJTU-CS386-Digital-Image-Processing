from GetWeight import get_weight
from Utils import get_image_stack, show_image
from Pyramid import *

# expand weight list to have rgb 3 channels
# it is very likely that numpy can handle it gracefully
# but I haven't found that method
def expand_weight(weight_list):
    shape = weight_list.shape
    expanded_weight_list = np.zeros((shape[0], shape[1], shape[2], 3))
    expanded_weight_list[:,:,:,0] = weight_list
    expanded_weight_list[:,:,:,1] = weight_list
    expanded_weight_list[:,:,:,2] = weight_list
    return expanded_weight_list


def exposure_fusion(image_list):
    weight_list = get_weight(image_list)
    expanded_weight_list = expand_weight(weight_list)
    fused_image = expanded_weight_list * image_list
    fused_image = np.sum(fused_image, axis=0)
    
    return fused_image

""" 1) construct gaussian pyramid of weights and laplacian pyramid of images,
    2) multiply them
    3) reconsturt output image from the laplacian pyramid
"""
def fusion(image_list, weight_list):
    # get weight pyramid
    weight_pyr_list = []
    for w in weight_list:
        gw = gauss_pyramid(w)
        weight_pyr_list.append(gw)

    for i in range(len(weight_pyr_list)):
        w_pyr = weight_pyr_list[i]
        w_pyr = w_pyr[0:-1]
        w_pyr.reverse()
        weight_pyr_list[i] = w_pyr

    # get image pyramid
    image_pyr_list = []
    for image in image_list:
        li = laplacian_pyramid(image)
        image_pyr_list.append(li)

    # fuse
    fused_pyr_list = []
    for w_pyr, i_pyr in zip(weight_pyr_list, image_pyr_list):
        fused_pyr_list.append(multi_pyr(w_pyr, i_pyr))
    fused_pyr = np.sum(fused_pyr_list, axis=0)
    
    # reconsturct
    fused_image = reconstruct_pyramid(fused_pyr)

    return fused_image


if __name__ == "__main__":
    image_list = get_image_stack("test_images")
    fused_image = exposure_fusion(image_list)
    fused_image = fused_image / 255.0

    show_image(fused_image, "fused image")
    cv2.imwrite("./result/result.jpg", fused_image * 255.0)

