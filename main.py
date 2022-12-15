"""
Function to detect the blobs in the given input image.
The python file is generated such as way that all the necessary helper functions needed to detect the blobs are created
together in this file in sequence.
"""
# Importing all the required libraries
import numpy as np
from convolution import conv2
import time
import cv2
import matplotlib.pyplot as plt

# Declaring Constants
initial_sigma = 1.6 # Intial Sigma Value
scale_factor = np.sqrt(2)  # Scaling factor to increase the scale
threshold = 0.005 # Threshold to detect blogs

# Laplacian of Gaussian Filter Generation
def log(sigma:float) -> np.ndarray:
    """
    Function to obtain Laplacian of Gaussian Filter with a given Variance.
    The function is computed by the formula of Laplacian Of Gaussian LoG(x,y)
    Args:
        - sigma: int/float
    Returns:
        - [np.ndarray]: Laplacian of Gaussian
    """
    w = (3*sigma + 0.5)
    s = int(2*w + 1)
    x = np.linspace(-w, w, s)
    y = np.linspace(-w, w, s)

    xspace, yspace = np.meshgrid(x, y)
    sigma_sq = sigma**2

    # Gaussian PDF generation
    _gaussianpdf = np.exp(-(xspace**2 + yspace**2)/(2*sigma_sq))
    num = (((xspace**2 + yspace**2) - 2*sigma_sq)/(sigma_sq*2)) * _gaussianpdf
    den = (np.pi*sigma_sq**2)
    kernel = num/den 
    return kernel

# Build Laplacian of Gaussian ScaleSpace
def laplacian_scalespace(input_img: np.array, sigmaCount: int) -> np.ndarray:
    """
    - Function creates a Laplacian Scalespace.
    - Filtering the image using the Convolution Function created in the previous project with arguments as image,
    padding and kernel type which gives the convolved image with scale-normalized Laplacian at current scale.
    - Getting square of Laplacian response for current level of scale space.
    Args:
        - input_img: np.ndarray
        - sigmaCount: int (takes the number of sigma for which the image is to be scaled and displayed 
        if sigmaCount = 10, there will be 10 values of sigma scaled with scaling factor)
        - scale_factor: scaling_factor (predefined)
    Returns:
        - [np.ndarray]: Laplacian ScaleSpace of an Image

    """ 
    # Scaling down the input image
    image_ = input_img/input_img.max()
    # Empty Scale Space
    laplacian_scale_space = []
    scale = initial_sigma
    for i in range(sigmaCount):
        kernel = log(scale)
        kernel = kernel * (scale**2)
        # Normalising the response
        conv_log_output = conv2(image_, kernel,0)
        # Square of the response
        sq_resp_output = conv_log_output ** 2
        laplacian_scale_space.append(sq_resp_output)
        scale = scale * scale_factor
    return laplacian_scale_space

# Non Max Suppression
def nms2d(scalespace: np.ndarray ,threshold = threshold) -> np.ndarray:
    """
    Function to get Non Maximum Suppression. This function is implemented to supress the values less than threshold.
    To check if the center value is greater than the neghbouring pixels around 3x3 Kernel, if yes set 1 else 0.
    Args:
        - input_img: Laplacian Scalespace Image Array
        - threshold: Threshold to get Blobs
    Returns:
        - List[np.ndarray]: List of filtered values above threshold converted to 1
    """
    rows,cols = scalespace.shape    
    # Initialising the Maximun
    image_maxima = np.zeros_like(scalespace)
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            each_max = np.max(scalespace[r-1:r+2,c-1:c+2])
            checkMaxima = np.logical_and(each_max == scalespace[r,c], each_max > threshold)
            if checkMaxima:
                image_maxima[r,c] = 1

    return image_maxima

def nms(img_maxima: np.ndarray, scalespaceLaplacian: np.ndarray) -> np.ndarray:
    """
    Function to get Non Maximum Suppression. This function is implemented to supress the values less than threshold.
    To check if the center value is greater than the pixels above and below around 3x3 Kernel, if yes set 1 else 0.
    Args:
        - input_img: Laplacian Scalespace Image Array
        - threshold: Threshold to get Blobs
    Returns:
        - List[np.ndarray]: List of filtered values above threshold converted to 1
    """
    imc = np.copy(img_maxima)
    ssl = np.copy(scalespaceLaplacian)
    for part in range(len(imc)):
        # if first part, only compare with part below
        if part == 0:
            maxima_points_2d = np.nonzero(imc[part])
            for i in range(maxima_points_2d[0].shape[0]):
                r = maxima_points_2d[0][i]
                c = maxima_points_2d[1][i]
                part_upper = part + 1
                part_lower = part - 1
                r_upper = r + 1
                r_lower = r - 1
                c_upper = c + 1
                c_lower = c - 1
                max_down = np.max(ssl[part_upper, r_lower:r_upper+1, c_lower:c_upper+1])
                if (ssl[part,r,c] < max_down):
                    imc[part,r,c] = 0

        # compare with only above part if last part
        if part == len(imc)-1:
            maxima_points_2d = np.nonzero(imc[part])
            if len(maxima_points_2d) == 1:
                return imc
            for i in range(maxima_points_2d[0].shape[0]):
                r = maxima_points_2d[0][i]
                c = maxima_points_2d[1][i]
                part_upper = part + 1
                part_lower = part - 1
                r_upper = r + 1
                r_lower = r - 1
                c_upper = c + 1
                c_lower = c - 1              
                max_up = np.max(ssl[part_lower, r_lower:r_upper+1, c_lower:c_upper+1])
                if (ssl[part,r,c] < max_up):
                    imc[part,r,c] = 0

        # parts in between two extremes
        else:
            maxima_points_2d = np.nonzero(imc[part])
            if len(maxima_points_2d) == 1:
                return imc
            for i in range(maxima_points_2d[0].shape[0]):
                r = maxima_points_2d[0][i]
                c = maxima_points_2d[1][i]
                part_upper = part + 1
                part_lower = part - 1
                r_upper = r + 1
                r_lower = r - 1
                c_upper = c + 1
                c_lower = c - 1
                max_up = np.max(ssl[part_lower, r_lower:r_upper+1, c_lower:c_upper+1])
                max_down = np.max(ssl[part_upper, r_lower:r_upper+1, c_lower:c_upper+1])
                if ((ssl[part,r,c] < max_down) or ssl[part,r,c] < max_up):
                    imc[part,r,c] = 0

    return imc

# Drawing Circles
def _drawcircles(img: np.ndarray,r:float,c:float,radius:float) -> np.ndarray:
    """
    Supporting function to draw the circles.
    Args:
        - img: Normalized Non Max Suppressed Input Laplacian ScaleSpace
        - r: Shape of Input Image Row (Normalized Laplacian ScaleSpace)
        - c: Shape of input Image Column (Normalized Laplacian ScaleSpace)
        - radius: Radius of the circle to be calculated
    Returns:
        - Image with Blobs
    """
    for x, y in zip(c,r):
        img = cv2.circle(img, (x,y),radius, (0,0,255))
    return img

def drawcircles(img: np.ndarray,nms_result:np.ndarray) -> np.ndarray:
    """
    Function to Draw the circles on the input image.
    Args:
        - img: Normalized Non Max Suppressed Input Laplacian ScaleSpace
        - nms_result: NonMaxSuppressed Result of the Laplacian ScaleSpace
    Returns:
        - Image with Blobs
    """
    if np.ndim(img) == 2 or img.shape[-1] == 1:
        img_ = img[:, :].copy()
        img_ = img_[:, :, None]
        draw_img = np.concatenate([img_, img_, img_], axis=2)
    else:
        draw_img = img.copy()

    scale = initial_sigma
    for map in nms_result:
        r,c = np.nonzero(map)
        radius = int(scale * scale_factor)
        draw_img = _drawcircles(draw_img,r,c,radius)
        scale = scale * scale_factor

    return draw_img
