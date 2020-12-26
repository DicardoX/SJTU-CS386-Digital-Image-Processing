from CRF import CRF_func, save_hdr
import cv2

# 获取HDR图像
#hdr = CRF_func()
hdr = cv2.imread("taipei.hdr", flags=cv2.IMREAD_ANYDEPTH)
HSVhdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV)

def size(image):
    return image.shape[1], image.shape[0], image.shape[1] * image.shape[0]

def gamma_correction(R, G, B):
    inner = (pow(R, 2.2) + pow(1.5 * G, 2.2) + pow(0.6 * B, 2.2)) / (1 + pow(1.5, 2.2) + pow(0.6, 2.2))
    return pow(inner, 1 / 2.2)

# 获取RGB分量
BlueVector = hdr[:, :, 0]
GreenVector = hdr[:, :, 1]
RedVector = hdr[:, :, 2]
# 获取尺寸信息
size_x, size_y, N = size(BlueVector)
# 获取亮度分量
Lw = HSVhdr[:, :, 2]

for i in range(0, size_x, 1):
    for j in range(0, size_y, 1):
        Lw[j][i] = gamma_correction(RedVector[j][i], GreenVector[j][i], BlueVector[j][i])

HSVhdr[:, :, 2] = Lw
hdr = cv2.cvtColor(HSVhdr, cv2.COLOR_HSV2BGR)
# 保存hdr
save_hdr(hdr, "hdr.jpg")




