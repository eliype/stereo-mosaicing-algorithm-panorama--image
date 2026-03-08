
import numpy as np
import scipy.signal as signal
import cv2



def lucas_kanade(first_frame, translated_frame):

    # first_frame = first_frame.convert('L')
    # translated_frame = translated_frame.convert('L')

    # first_frame = np.array(first_frame)
    # translated_frame = np.array(translated_frame)

    # first_frame = blur(first_frame)
    # translated_frame = blur(translated_frame)

    lucas_kanade_matrix, x_translated_derivative, y_translated_derivative = lucas_matrix(translated_frame)
    B = lucas_B(first_frame,translated_frame,x_translated_derivative,y_translated_derivative)

    # u - motion in x
    # v - motion in y

    u,v = 0,0
    alpha = 0
    for i in range(3):

        solution = np.linalg.solve(lucas_kanade_matrix, B)
        du, dv = solution

        if np.sqrt(du**2 + dv**2) < 0.04:
            return u,v

        # update motions
        u += du[0]
        v += dv[0]
        # alpha += alpha do not touch

        translated_frame_calc = affine_transformation(translated_frame, -v, -u,0)
        B = lucas_B(first_frame, translated_frame_calc, x_translated_derivative, y_translated_derivative)

    return u,v

def affine_transformation(frame,v,u,alpha):

    theta = alpha
    alpha = np.cos(theta)
    beta = np.sin(theta)
    center = (frame.shape[1]/2.0, frame.shape[0]/2.0)


    M = np.array([
        [alpha, -beta, (1 - alpha) * center[0] + beta * center[1] + u],
        [beta, alpha, -beta * center[0] + (1 - alpha) * center[1] + v]
    ], dtype=np.float32)

    transformed_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)#borderMode=cv2.BORDER_REFLECT

    return transformed_frame


def lucas_B(first_frame, translated_frame,x_translated_derivative, y_translated_derivative):


    ##### compute B solution matrix

    B = np.zeros((2,1))
    It = translated_frame - first_frame

    y_grid, x_grid = np.mgrid[0:translated_frame.shape[0], 0:translated_frame.shape[1]]
    Ix = x_translated_derivative
    Iy = y_translated_derivative


    B[0,0] = np.sum(Ix * It)*-1
    B[1,0] = np.sum(Iy * It)*-1
    # B[2,0] = np.sum((Iy * x_grid -Ix * y_grid)*It)*-1

    return B


def lucas_matrix(translated_frame):

    ##### creating parameters

    lucas_kanade_matrix = np.zeros((2,2))
    y_grid,x_grid = np.mgrid[0:translated_frame.shape[0], 0:translated_frame.shape[1]]

    # x derivative
    Ix = x_derivative(translated_frame)

    # y derivative
    Iy = y_derivative(translated_frame)

    # x*x
    lucas_kanade_matrix[0,0] = np.sum(Ix*Ix)

    # y*y
    lucas_kanade_matrix[1,1] = np.sum(Iy*Iy)

    # x*y
    lucas_kanade_matrix[0,1] = np.sum(Ix*Iy)

    # y*x
    lucas_kanade_matrix[1,0] = lucas_kanade_matrix[0,1]

    # ###
    # lucas_kanade_matrix[0,2] = np.sum(Iy*Ix*x_grid - Ix*Ix*y_grid)
    # ###
    # lucas_kanade_matrix[1,2] = np.sum(Iy*Iy*x_grid - Ix*Iy*y_grid)
    # ###
    # lucas_kanade_matrix[2,0] = np.sum(Ix*(Iy*x_grid - Ix*y_grid))
    # ###
    # lucas_kanade_matrix[2,1] = np.sum(Iy*(Iy*x_grid - Ix*y_grid))
    # ###
    # lucas_kanade_matrix[2,2] = np.sum(np.pow(Iy*x_grid - Ix*y_grid,2))

    return lucas_kanade_matrix,Ix,Iy

def blur(frame):
    gaussian_blur = np.array([
        [1, 4, 7,10,7, 4, 1], ], dtype=np.float32)
    gaussian_blur = gaussian_blur / gaussian_blur.sum()

    frame = signal.convolve2d(frame, gaussian_blur, mode='same', fillvalue=0)

    gaussian_blur = gaussian_blur.reshape((-1, 1))
    frame = signal.convolve2d(frame, gaussian_blur, mode='same', fillvalue=0)

    return frame.astype(np.float32) / 255.0
def x_derivative(frame):

    sobel_kernel = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]) * 1/8

    return signal.convolve2d(frame, sobel_kernel, mode='same',fillvalue=0)

def y_derivative(frame):

    sobel_kernel = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) * 1/8

    return signal.convolve2d(frame, sobel_kernel, mode='same',fillvalue=0)







