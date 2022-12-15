# Function to implement the Convolution of the Image with 4 types of padding and kernels as required.
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math

# Padding Function
def add_padding(f: np.array,wr: int,wc: int, pad: int) -> np.array:
    """
    Adds padding to the Input Matrix
    Args:
    1) Input Image -> f
    2) Kernel Size -> w
    3) Padding type -> 0 for clip/zero-padding (Default)
                       1 for wrap padding
                       2 for copy edge
                       3 for reflect across edge
    Returns:
    Padded Matrix (np.array)
    """
    fr = f.shape[0]
    fc = f.shape[1]
    #Padding around the edges according to size of Kernel
    topmost= int(math.ceil((wr-1)/2))
    downmost = int(math.floor((wr-1)/2))
    leftmost = int(math.ceil((wc-1)/2))
    rightmost = int(math.floor((wc-1)/2))
    
    #Zero Padding
    extended_padding = (fr + topmost + downmost, fc + leftmost + rightmost)
    padded_image = np.zeros(extended_padding)
    #Fitting original image into zero padded matrix 
    padded_image[topmost : topmost + fr, leftmost : leftmost + fc] = f
    
    #Warp
    if pad == 1:
        if topmost != 0:
            padded_image[0:topmost,:]=padded_image[-1*(topmost + downmost):topmost+f.shape[0],:]
        if downmost != 0:
            padded_image[-1*(downmost) : , : ] = padded_image[topmost : topmost + downmost, :]
        if rightmost != 0:
            padded_image[ : ,-1*(rightmost) : ] = padded_image[ : ,leftmost : leftmost + rightmost]
        if leftmost != 0:
            padded_image[ : ,0 : leftmost] = padded_image[ : ,-1*(leftmost+rightmost) : leftmost + f.shape[1]]
            
    #Copy Edge        
    elif pad == 2:
        if topmost != 0: 
            padded_image[0 : topmost, : ] = padded_image[[topmost], : ]
        if downmost != 0: 
            padded_image[-1*(downmost): , : ] = padded_image[[-1*downmost-1], :]
        if rightmost != 0: 
            padded_image[ : ,-1*(rightmost) : ] = padded_image[ : ,[-1*(rightmost)-1]]
        if leftmost != 0: 
            padded_image[ : ,0 : leftmost] = padded_image[ : ,[leftmost]]      
    #        
    elif pad == 3:
        if topmost != 0:
            padded_image[0 : topmost, : ] = np.flip(padded_image[topmost : 2*topmost, :],axis = 0)
        if downmost != 0:
            padded_image[-1*(downmost) : , : ] = np.flip(padded_image[-2*(downmost) : -1*(downmost), : ], axis = 0)
        if rightmost != 0:
            padded_image[ : ,-1*(rightmost) : ] = np.flip(padded_image[ : , -2*(rightmost) : -1*(rightmost)], axis = 1)
        if leftmost != 0:
            padded_image[ : ,0 : leftmost] = np.flip(padded_image[ : ,leftmost : 2* leftmost],axis = 1)
            
    return padded_image


# Convolution Function
def conv2(f: np.array,w: np.array,pad: int) -> np.ndarray:
    """
    Adds padding to the Input Matrix
    Args:
    1) Input Image -> f
    2) Kernel -> w
    3) Padding type -> 0 for clip/zero-padding (Default)
                       1 for wrap padding
                       2 for copy edge
                       3 for reflect across edge
    Returns:
    Convoluted Image (np.array)
    
    Gray Image is a two component Image. In case of three component images i.e. RGB, we need to split each component 
    and apply convolution seperately. Once the convolution is applied, we will merge the components together and return
    the convoluted RGB image.
    """    
    # FOR GRAY IMAGES
    if len(f.shape) < 3:
        f_padded = add_padding(f,w.shape[0],w.shape[1], pad)
        convolved_matrix = np.zeros((f.shape[0], f.shape[1]))
        for r in range(convolved_matrix.shape[0]):
            for c in range(convolved_matrix.shape[1]):
                convolved_matrix[r][c]= np.sum(np.multiply(f_padded[r:r+w.shape[0],c:c+w.shape[1]],w))
            
    # FOR RGB IMAGES
    elif len(f.shape) == 3:
        
        b,g,r = cv2.split(f)
        
        fb_padded = add_padding(b, w.shape[0], w.shape[1], pad) 
        fg_padded = add_padding(g, w.shape[0], w.shape[1], pad) 
        fr_padded = add_padding(r, w.shape[0], w.shape[1], pad)
        
        convolved_bmatrix = np.zeros((b.shape[0],b.shape[1]))
        convolved_gmatrix = np.zeros((g.shape[0],g.shape[1]))
        convolved_rmatrix = np.zeros((r.shape[0],r.shape[1]))
        
        for r in range(convolved_bmatrix.shape[0]):
            
            for c in range(convolved_bmatrix.shape[1]):
                convolved_bmatrix[r][c]= np.sum(np.multiply(fb_padded[r:r+w.shape[0],c:c+w.shape[1]],w))
                convolved_gmatrix[r][c]= np.sum(np.multiply(fg_padded[r:r+w.shape[0],c:c+w.shape[1]],w))
                convolved_rmatrix[r][c]= np.sum(np.multiply(fr_padded[r:r+w.shape[0],c:c+w.shape[1]],w))
                
        convolved_matrix = cv2.merge((convolved_bmatrix,convolved_gmatrix,convolved_rmatrix)).astype(np.uint8)
    else:
        print("\nInput out of bounds")
            
    return convolved_matrix
