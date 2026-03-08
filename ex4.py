from PIL import Image
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mediapy as media
import cv2
import os


#################################################
#                                               #
#               transformation                  #
#                                               #
#################################################

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
        du, dv,alpha = solution

        if np.sqrt(du**2 + dv**2) < 0.04:
            return u,v,alpha

        # update motions
        u += du[0]
        v += dv[0]
        # alpha += alpha do not touch

        translated_frame_calc = affine_transformation(translated_frame, -v, -u,-alpha)
        B = lucas_B(first_frame, translated_frame_calc, x_translated_derivative, y_translated_derivative)

    return u,v,alpha

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

    B = np.zeros((3,1))
    It = translated_frame - first_frame

    y_grid, x_grid = np.mgrid[0:translated_frame.shape[0], 0:translated_frame.shape[1]]
    Ix = x_translated_derivative
    Iy = y_translated_derivative


    B[0,0] = np.sum(Ix * It)*-1
    B[1,0] = np.sum(Iy * It)*-1
    B[2,0] = np.sum((Iy * x_grid -Ix * y_grid)*It)*-1

    return B


def lucas_matrix(translated_frame):

    ##### creating parameters

    lucas_kanade_matrix = np.zeros((3,3))
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

    ###
    lucas_kanade_matrix[0,2] = np.sum(Iy*Ix*x_grid - Ix*Ix*y_grid)
    ###
    lucas_kanade_matrix[1,2] = np.sum(Iy*Iy*x_grid - Ix*Iy*y_grid)
    ###
    lucas_kanade_matrix[2,0] = np.sum(Ix*(Iy*x_grid - Ix*y_grid))
    ###
    lucas_kanade_matrix[2,1] = np.sum(Iy*(Iy*x_grid - Ix*y_grid))
    ###
    lucas_kanade_matrix[2,2] = np.sum(np.pow(Iy*x_grid - Ix*y_grid,2))

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


#################################################
#                                               #
#           multiple LK algorithm               #
#                                               #
#################################################

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

    gaus_pyramid = [blur(image)]

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

        trans_current = affine_transformation(trans_gaus_pyramid[i],-v,-u,-alpha)

        du,dv,alpha = lucas_kanade(first_gaus_pyramid[i], trans_current)

        if is_alpha:
            true_alpha = alpha
            is_alpha = False
        v += dv
        u += du
        # print(f'ver:{u:2f},hor:{v:2f},angel:{true_alpha[0]:3f}')
        if i > 0:
            v *= 2
            u *= 2


    return u,v,true_alpha[0]


def real_blur(image):
    # img = Image.fromarray(image)
    blurred = cv2.GaussianBlur(image,(5,5),0)

    return np.array(blurred)

#################################################
#                                               #
#              creating mosaic                  #
#                                               #
#################################################

def mosaic_transformation(frame,v,u,alpha):

    theta = alpha
    alpha = np.cos(theta)
    beta = np.sin(theta)
    center = (frame.shape[1]/2.0, frame.shape[0]/2.0)


    M = np.array([
        [alpha, -beta, (1 - alpha) * center[0] + beta * center[1] + u],
        [beta, alpha, -beta * center[0] + (1 - alpha) * center[1] + v]
    ], dtype=np.float32)

    transformed_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR
                                       )#borderMode=cv2.BORDER_REFLECT

    return transformed_frame

def align_frames(video_test,real_video,extended):

    video = []
    canvas_height = 0
    v_from_origin = 0
    canvas_width = 0
    save_frame = real_video[0]
    for i in range(1, len(video_test)):#len(video_test)
        # print(i)

        o_second_frame = np.array(real_video[i])
        first_frame = np.array(Image.fromarray(video_test[i - 1]).convert('L'))
        second_frame = np.array(Image.fromarray(video_test[i]).convert('L'))

        u, v, angel = lucas_kanade_algorithm(first_frame, second_frame)
        # print(f'u:{u},v:{v},angel:{angel}')

        v_from_origin += v #chancge to np.round(v)

        if v_from_origin > canvas_height:
            canvas_height = v_from_origin
        canvas_width += int(np.abs(u)+extended)

        red_frame = mosaic_transformation(o_second_frame[:,:,0], -v_from_origin, 0, -angel)
        greed_frame = mosaic_transformation(o_second_frame[:,:,1], -v_from_origin, 0, -angel)
        blue_frame = mosaic_transformation(o_second_frame[:,:,2], -v_from_origin, 0, -angel)
        rgb_frame = np.stack([red_frame, greed_frame, blue_frame],axis = -1)

        video.append([save_frame,int(np.abs(u)+extended)])
        save_frame = rgb_frame

    video.append([save_frame,0])

    # print(f'max_v:{canvas_height}')
    # print(f'canvas_size:{canvas_width}')

    return video, int(canvas_width), int(canvas_height)


def creating_canvas(canvas_width, canvas_height):

    return np.zeros((canvas_height, canvas_width,3), np.float32)


def creating_mosaic(video,stripe_p,canw,canh,middle_canvas):
    canvas = creating_canvas(canw, middle_canvas + canh)
    current = 0

    # print(stripe_p)
    for i in range(len(video) - 1):  # len(video)-1
        stripe_width = np.round(video[i][1])
        canvas[int(canh / 2):middle_canvas + int(canh / 2), current:current + stripe_width] = video[i][0][
            :, stripe_p :stripe_p + stripe_width]
        current = current + stripe_width

    return canvas

def mosaic_video(video_test,n_out_frames,extended=0):
    middle = video_test.shape[1]
    video, canw, canh = align_frames(video_test[:,middle // 2: int(middle * (3/4))],video_test,extended)

    frames = []
    skips = (video_test.shape[2])//n_out_frames
    for stripe_p in range(10,video_test.shape[2]-20,10): #range(10,video_test.shape[2] - 10,10) ###### do not forget to change 20 to 10

        mosaic_frame = creating_mosaic(video,stripe_p,canw,canh,middle)
        frames.append(mosaic_frame)

    return frames

def generate_panorama(input_frames_path, n_out_frames):

    # frame_files = sorted([f for f in os.listdir(input_frames_path) if f.endswith('.jpg')])
    # frames = []
    # for frame_file in frame_files:
    #     img_path = os.path.join(input_frames_path, frame_file)
    #     img = Image.open(img_path)
    #     frames.append(img)

    frames = np.array(media.read_video("Example inputs/bad video.mp4"))

    panorama_frames = mosaic_video(frames,n_out_frames)
    panorama_frames = np.array(panorama_frames).astype(np.uint8)

    panorama_pil = []
    for img in panorama_frames:
        image_PIL = Image.fromarray(img)
        panorama_pil.append(image_PIL)

    for img in panorama_pil:
        plt.imshow(img)
        plt.show()

    frames_resized = []
    for frame in panorama_frames:
        # Resize to more standard dimensions
        resized = cv2.resize(frame, (800, 272))  # Keep height, reduce width
        frames_resized.append(resized)

    video_rgb = np.array(frames_resized)

    media.write_video('Example outputs/video_garden_output.mp4', video_rgb, fps=24)
    return panorama_pil


generate_panorama("Example inputs/boat.mp4",5)