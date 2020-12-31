import math
import numpy as np

# max pixel vaule
N = 255

# l is lamdba, the constant that determines the amount of smoothness
L = 50

def hdr_debvec(img_list, exposure_times, number_of_samples_per_dimension=20):
    B = [math.log(e, 2) for e in exposure_times]
    l = L
    w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]

    samples = []
    width = img_list[0].shape[0]
    height = img_list[0].shape[1]
    width_iteration = width / number_of_samples_per_dimension
    height_iteration = height / number_of_samples_per_dimension

    w_iter = 0
    h_iter = 0

    Z = np.zeros((len(img_list), number_of_samples_per_dimension * number_of_samples_per_dimension))
    for img_index, img in enumerate(img_list):
        h_iter = 0
        for i in range(number_of_samples_per_dimension):
            w_iter = 0
            for j in range(number_of_samples_per_dimension):
                if math.floor(w_iter) < width and math.floor(h_iter) < height:
                    pixel = img[math.floor(w_iter), math.floor(h_iter)]
                    Z[img_index, i * j] = pixel
                w_iter += width_iteration
            h_iter += height_iteration

    return response_curve_solver(Z, B, l, w)


# Implementation of paper's Equation(3) with weight
def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0) * np.size(Z, 1) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = int(Z[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n + i] = -wij
            b[k] = wij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n - 1):
        A[k][i] = l * w[i + 1]
        A[k][i + 1] = -2 * l * w[i + 1]
        A[k][i + 2] = l * w[i + 1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


# In[5]:

# Implementation of paper's Equation(6)
def construct_radiance_map(g, Z, ln_t, w):
    acc_E = [0] * len(Z[0])
    ln_E = [0] * len(Z[0])

    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        acc_w = 0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z] * (g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i] / acc_w if acc_w > 0 else acc_E[i]
        acc_w = 0

    return ln_E


def construct_hdr(img_list, response_curve, exposure_times):
    # Construct radiance map for each channels
    img_size = img_list[0][0].shape
    w = [z if z <= 0.5 * 255 else 255 - z for z in range(256)]
    ln_t = np.log2(exposure_times)

    vfunc = np.vectorize(lambda x: math.exp(x))
    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')

    # construct radiance map for BGR channels
    for i in range(3):
        print(' - Constructing radiance map for {0} channel .... '.format('BGR'[i]), end='', flush=True)
        Z = [img.flatten().tolist() for img in img_list[i]]
        E = construct_radiance_map(response_curve[i], Z, ln_t, w)
        # Exponational each channels and reshape to 2D-matrix
        hdr[..., i] = np.reshape(vfunc(E), img_size)
        print('done')

    return hdr


def process(images, ExposureTimes):

    img_list_b, exposure_times = np.array(images)[:,:,:,0], ExposureTimes.copy()
    img_list_g, exposure_times = np.array(images)[:,:,:,1], ExposureTimes.copy()
    img_list_r, exposure_times = np.array(images)[:,:,:,2], ExposureTimes.copy()

    # Solving response curves
    gb, _ = hdr_debvec(img_list_b, exposure_times)
    gg, _ = hdr_debvec(img_list_g, exposure_times)
    gr, _ = hdr_debvec(img_list_r, exposure_times)

    # Show response curve

    hdr = construct_hdr([img_list_b, img_list_g, img_list_r], [gb, gg, gr], exposure_times)

    return hdr



