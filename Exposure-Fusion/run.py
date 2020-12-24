from Utils import get_image_stack, show_image
from GetWeight import get_weight
from ExposureFusion import fusion
import cv2

if __name__ == "__main__":
    image_list = get_image_stack("./test_images")
    
    weight_list = get_weight(image_list)
    fused_image = fusion(image_list, weight_list)
    
    fused_image = fused_image / 255.0
    show_image(fused_image, "fused image")
    cv2.imwrite("./result/result.jpg", fused_image * 255.0)
