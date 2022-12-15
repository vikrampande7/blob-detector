import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from convolution import conv2
from main import log, laplacian_scalespace, nms2d, nms, _drawcircles, drawcircles
threshold = 0.005

def run_blob_detector(image: np.ndarray,sigma:int, isGrey:int) -> np.ndarray:
    """
    Args:
        - image: Input Image
        - sigma: Number of Variance
        - isGrey: if 1 -> Grey Image else RGB
    Returns:
        - Output Image with Blobs

    """
# Reading the Input Image
    img = cv2.imread(image) # Reading the Image
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if isGrey == 0:
        img = img
    if isGrey == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grey Image Conversion

    # Laplacian of Gaussian Function and Laplacian Scale Space
    get_log = laplacian_scalespace(img_grey,sigma)
    get_log_copy = get_log.copy()
    for i in range(len(get_log)):
        nms_points = nms2d(get_log[i], threshold)
        get_log_copy[i] = nms_points

    # Non-Max-Suppression
    nms_final = nms(get_log_copy,get_log)

    # Draw Circles and Plot
    draw_final = drawcircles(img,nms_final)
    # print(draw_final.shape)
    # plt.imshow(draw_final)
    # plt.show()

    return draw_final    



def get_resultANDTime():
    """
    This function will return the time taken for the operation for blob detection and save all the result images ( RGB + Grey ) with various variances.
    """
    start_time = time.time()

    # blobs_lena = run_blob_detector('TestImages4Project/own_lena.png',3,1)
    # cv2.imwrite('ResultImages/blobs_lena.png',blobs_lena)

    blobs_butterfly = run_blob_detector('TestImages4Project/butterfly.jpg',7,0)
    cv2.imwrite('ResultImages/blobs_butterfly.png',blobs_butterfly)

    blobs_einstein = run_blob_detector('TestImages4Project/einstein.jpg',3,1)
    cv2.imwrite('ResultImages/blobs_einstein.png',blobs_einstein)

    blobs_fishes = run_blob_detector('TestImages4Project/fishes.jpg',5,1)
    cv2.imwrite('ResultImages/blobs_fishes.png',blobs_fishes)

    blobs_sunflowers = run_blob_detector('TestImages4Project/sunflowers.jpg',7,0)
    cv2.imwrite('ResultImages/blobs_sunflowers.png',blobs_sunflowers)

    blobs_tiger = run_blob_detector('TestImages4Project/own_tiger.jpg',5,0)
    cv2.imwrite('ResultImages/blobs_tiger.png',blobs_tiger)

    blobs_naruto = run_blob_detector('TestImages4Project/own_naruto.jpg',5,0)
    cv2.imwrite('ResultImages/blobs_naruto.png',blobs_naruto)

    blobs_road = run_blob_detector('TestImages4Project/own_road.jpeg',5,0)
    cv2.imwrite('ResultImages/blobs_road.png',blobs_road)

    blobs_earth = run_blob_detector('TestImages4Project/own_earth.png',3,0)
    cv2.imwrite('ResultImages/blobs_earth.png',blobs_earth)

    #time_taken = float(time.time()) - float(start_time)

    #print(time_taken)
    print('\nTime taken to process 8 images is ', time.time() - start_time, ' seconds.')
    #print("Time taken to process 8 images is " + time_taken + " seconds." )

    return True


get_resultANDTime()