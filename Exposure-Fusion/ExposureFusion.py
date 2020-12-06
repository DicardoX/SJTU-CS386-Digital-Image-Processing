from GetWeight import *

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


# """ only fuse intensity channel"""
# def var_exposure_fusion(image_list):
#     weight_list = get_weight(image_list)
#     yuv_image_list = [rgb2yuv(image) for image in image_list]
#     yuv_image_list = np.array(yuv_image_list)

#     intensity_list = yuv_image_list[:, :, :, 0]
#     intensity_list = intensity_list / 255.0
#     fused_intensity = intensity_list * weight_list
#     fused_intensity = np.sum(fused_intensity, axis=0) * 255.0
    
#     fused_color = yuv_image_list[1, :, :, 1:3]

#     shape = fused_intensity.shape
#     fused_image = np.zeros((shape[0], shape[1], 3))
#     fused_image[:, :, 0] = fused_intensity
#     fused_image[:, :, 1:3] = fused_color

#     fused_image = fused_image.astype(np.uint8)
#     fused_image = yuv2rgb(fused_image)
#     return fused_image


if __name__ == "__main__":
    image_list = get_image_stack("test_images")
    fused_image = exposure_fusion(image_list)
    fused_image = fused_image / 255.0

    show_image(fused_image, "fused image")
    cv2.imwrite("./result/result.jpg", fused_image * 255.0)

