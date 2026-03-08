import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.signal as signal
import transformation as ts


def reduce(image):

    np_image = np.array(image)
    # Gaussian blend kernel
    kernel = np.array([[1, 4, 6, 4, 1]])
    kernel = kernel / kernel.sum()
    #
    blurred_image = signal.convolve2d(np_image, kernel, mode='same', fillvalue=0)
    # Gaussian blend kernel transpose
    kernel = kernel.reshape((-1,1))

    blurred_image = signal.convolve2d(blurred_image, kernel, mode='same', fillvalue=0)
    # subsample the image in y and x direction
    np_image = np.array(blurred_image)[::2,::2]

    return np_image

def gaussian_pyramid(image,levels):

    gaus_pyramid = [ts.blur(image)]

    for i in range(levels):
        image = reduce(image)
        gaus_pyramid.append(image.astype(np.float32)/255.0)

    return gaus_pyramid

def lucas_kanade_algorithm(first_frame, translated_frame):

    first_gaus_pyramid = gaussian_pyramid(first_frame, 5)
    trans_gaus_pyramid = gaussian_pyramid(translated_frame, 5)

    v,u,alpha = 0,0,0
    true_alpha = 0
    is_alpha = True
    for i in range(len(first_gaus_pyramid)-1,-1,-1):

        trans_current = ts.affine_transformation(trans_gaus_pyramid[i],-v,-u,-alpha)

        du,dv = ts.lucas_kanade(first_gaus_pyramid[i], trans_current)


        v += dv
        u += du

        if i > 0:
            v *= 2
            u *= 2
        if is_alpha:
            # true_alpha = alpha
            is_alpha = False

    return u,v  #,true_alpha[0]


def real_blur(image):
    # img = Image.fromarray(image)
    blurred = cv2.GaussianBlur(image,(5,5),0)

    return np.array(blurred)











