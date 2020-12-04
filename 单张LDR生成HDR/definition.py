import numpy as np
import cv2
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

# Selective Reflectance Scaling. Stretch the reflectance of pixel whose illuminance brighter than mean value
def SRS(reflectance, illuminace):
    gamma_R = 0.5   # constant
    def piecewise_func(R, I, meanI):      # Definition of the piecewise function
        return R * (I / meanI) ** gamma_R if I > meanI else R
    mean_I = np.mean(illuminace)
    srs_func = np.vectorize(piecewise_func)         # Since the input of the piecewise function is vector
    result = srs_func(reflectance, illuminace, mean_I)
    return result

# Weighted Least Square Filter
def WLSF(IN, Lambda=1.0, Alpha=1.0):
    # IN: Input image (2D grayscale image, type float)
    # Lambda: Balances between the data term and the smoothness term. Increasing lambda will produce smoother images.
    # Alpha: Gives a degree of control over the affinities by non-lineary scaling the gradients. Increasing alpha will result in sharper preserved edges.

    L = np.log(IN + 1e-22)  # Source image for the affinity matrix. log_e(IN)
    smallNum = 1e-6  # To prevent division by zero
    height, width = IN.shape  # Height and width of the input image
    N = height * width  # Number of pixels

    # Compute affinities between adjacent pixels based on gradients of L
    dy = np.diff(L, n=1, axis=0)  # axis=0 is vertical direction
    dy = -Lambda / (np.abs(dy) ** Alpha + smallNum)
    dy = np.pad(dy, ((0, 1), (0, 0)), 'constant')  # Add zeros row
    dy = dy.flatten(order='F')  # Reduce dimension by column

    dx = np.diff(L, n=1, axis=1)  # axis=1 is horizontal direction
    dx = -Lambda / (np.abs(dx) ** Alpha + smallNum)
    dx = np.pad(dx, ((0, 0), (0, 1)), 'constant')  # Add zeros col
    dx = dx.flatten(order='F')  # Reduce dimension by column

    # Construct matrix
    B = np.concatenate([[dx], [dy]], axis=0)  # Combine array
    d = np.array([-height, -1])

    A = spdiags(B, d, N, N)  # Return a sparse matrix from diagonals

    e = dx
    w = np.pad(dx, (height, 0), 'constant')
    w = w[0:-height]
    s = dy
    n = np.pad(dy, (1, 0), 'constant')
    n = n[0:-1]

    D = 1.0 - (e + w + s + n)

    A = A + A.transpose() + spdiags(D, 0, N, N)  # Transpose() is the matrix transpose function

    # Solve
    OUT = spsolve(A, IN.flatten(order='F'))
    return np.reshape(OUT, (height, width), order='F')

# Scale function, in the form of sigmoid, used for VIG
def scale_func(v_, mean_I_, max_I_):
    r = 1.0 - mean_I_ / max_I_
    sigma_s = 1.0
    f = lambda v: r * (1 / (1 + np.exp(-sigma_s * (v - mean_I_))) - 0.5)

    fv_ = [f(vk) for vk in v_]
    return fv_

# Virtual Illumination Generation. Generation of virtual exposure images(5 levels)
def VIG(illuminace, inverse_illuminace):
    inverse_illuminace /= np.max(inverse_illuminace)    # Normalization
    meanI = np.mean(illuminace)     # Mean illuminance
    maxI = np.max(illuminace)       # Max illuminance
    ####### According to the definition in the paper #######
    v1 = 0.2
    v3 = meanI
    v2 = 0.5 * (v1 + v3)
    v5 = 0.8
    v4 = 0.5 * (v3 + v5)
    v = [v1, v2, v3, v4, v5]
    ####### End here #######################################
    fv_list_ = scale_func(v, meanI, maxI)
    # equation (7)
    # I_k = [(1 + fv) * illuminace for fv in fvk_list_]
    # equation (8)
    I_k_ = [(1 + fv) * (illuminace + fv * inverse_illuminace) for fv in fv_list_]

    return I_k_

# Fusion of multiple exposure images. Generate pseudo multi-exposure luminances from the enhanced reflectance R′ and the Ik.
def tone_reproduction(bgr_image, L, R_, Ik_list_, FLAG):
    Lk_list_ = [ np.exp(R_) * Ik for Ik in Ik_list_ ]
    L = L + 1e-22

    gamma_t = 1.0
    b, g, r = cv2.split(bgr_image)
    # Restore color image
    if FLAG == False:                           # Simple map
        Sk_list_ = [cv2.merge((Lk * (B / L) ** gamma_t, Lk * (G / L) ** gamma_t, Lk * (R / L) ** gamma_t)) for Lk in Lk_list_]
        return Sk_list_[2]
    else:                                       # Weighted map
        Wk_list = []
        for index, Ik in enumerate(Ik_list_):
            if index < 3:
                wk = Ik / np.max(Ik)
            else:
                temp = 0.5 * (1 - Ik)
                wk = temp / np.max(temp)
            Wk_list.append(wk)

        A = np.zeros_like(Wk_list[0])
        B = np.zeros_like(Wk_list[0])
        for lk, wk in zip(Lk_list_, Wk_list):   # Weighted sum
            A = A + lk * wk
            B = B + wk

        L_ = (A / B)
        ratio = np.clip(L_ / L, 0, 3)   # Clip unreasonable values
        b_ = ratio * b
        g_ = ratio * g
        r_ = ratio * r
        out = cv2.merge( ( b_, g_, r_ ) )
        return np.clip(out, 0.0, 1.0)

# Main function class
class HDR():
    def __init__(self, flag):
        self.weighted_fusion = flag
        self.wls = WLSF
        self.srs = SRS
        self.vig = VIG
        self.tonemap = tone_reproduction

    def process(self, image):
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0     # Color space conversion
        image = 1.0 * image / 255
        L = 1.0 * S                                     # Calculate the input luminance

        I = self.wls(S)                                 # Obtain estimated illumination
        R = np.log(L + 1e-22) - np.log(I + 1e-22)       # Obtain reflectance information
        R_enhanced = self.srs(R, L)                     # Obtain enhanced reflectance
        virtual_I_K_ = self.vig(L, 1.0 - L)                      # Obtain list of virtual generated illumination

        result_ = self.tonemap(image, L, R_enhanced, virtual_I_K_, self.weighted_fusion)
        return result_

