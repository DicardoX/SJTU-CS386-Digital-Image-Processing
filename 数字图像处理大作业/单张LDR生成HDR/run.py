import numpy as np
import cv2
from matplotlib import pyplot as plt
from definition import HDR
import math
from PIL import Image
from pylab import *
import skimage
from skimage import metrics

# Reshape image
def reshape_image(image):
    shape = image.shape
    if shape[0] <= 1024 and shape[1] <= 1024:
        return image
    print("Original shape: ", shape)
    while shape[0] > 1024 or shape[1] > 1024:
        image = cv2.resize(image, (int(shape[1] / 2), int(shape[0] / 2)), interpolation=cv2.INTER_AREA)
        shape = image.shape
    print("Current shape: ", image.shape)
    return image

# Draw image
def draw_image(image):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.flip(image, 2))
    plt.title('Output Image')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Input filename and read the image
    filename = "./test_image/" + input("Please input the filename of the picture: ")
    image = cv2.imread(filename)

    # If failed to read image
    if image is None:
        print ("Error occurred when reading the image!")
        exit(0)


    # Reshape the size of image, in case that the input image is too big
    image = reshape_image(image)
    # Add hdr filter. True:  Using weighted fusion;  False: Average fusion.
    HDR_Filer = HDR(True)
    print("Processing...")
    output_image = HDR_Filer.process(image) # output_image is a 0-1 matrix

    # Output the result
    cv2.imwrite('./result/result.jpg', 255 * output_image)

    # Draw image
    draw_image(image=output_image)

    ################ Evaluation Part ########################
    # PSNR, Peak Signal to Noise Ratio
    def psnr(img1, img2):
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    psnrValue = psnr(output_image, image)

    # SSIM, Structural Similarity
    img1 = cv2.cvtColor(uint8(output_image), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ssimScore = skimage.metrics.structural_similarity(img1, img2, data_range=255, multichannel=True)

    print("PSNR score:", psnrValue, "|", "SSIM score:", ssimScore)
    ################### End here ############################

    # Print usage message
    print("-----------------------------------------------------")
    print("Please enter /MyHDR/result/ to obtain the 'result.jpg'")
