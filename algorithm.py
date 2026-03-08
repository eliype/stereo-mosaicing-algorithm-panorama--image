from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import LK
import cv2
import transformation as ts



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
        print(i)

        o_second_frame = np.array(real_video[i])
        first_frame = np.array(Image.fromarray(video_test[i - 1]).convert('L'))
        second_frame = np.array(Image.fromarray(video_test[i]).convert('L'))

        u, v= LK.lucas_kanade_algorithm(first_frame, second_frame)
        # print(f'u:{u},v:{v},angel:{angel}')

        v_from_origin += v #chancge to np.round(v)

        if v_from_origin > canvas_height:
            canvas_height = v_from_origin
        canvas_width += int(np.abs(u)+extended)

        red_frame = mosaic_transformation(o_second_frame[:,:,0], -v_from_origin, 0, 0)
        greed_frame = mosaic_transformation(o_second_frame[:,:,1], -v_from_origin, 0, 0)
        blue_frame = mosaic_transformation(o_second_frame[:,:,2], -v_from_origin, 0, 0)
        rgb_frame = np.stack([red_frame, greed_frame, blue_frame],axis = -1)

        video.append([save_frame,int(np.abs(u)+extended)])
        save_frame = rgb_frame

    video.append([save_frame,0])

    print(f'max_v:{canvas_height}')
    print(f'canvas_size:{canvas_width}')

    return video, int(canvas_width), int(canvas_height)


def creating_canvas(canvas_width, canvas_height):

    return np.zeros((canvas_height, canvas_width,3), np.float32)


def creating_mosaic(video,stripe_p,canw,canh,middle_canvas):
    canvas = creating_canvas(canw, middle_canvas + canh)
    current = 0

    print(stripe_p)
    for i in range(len(video) - 1):  # len(video)-1
        stripe_width = np.round(video[i][1])
        canvas[int(canh / 2):middle_canvas + int(canh / 2), current:current + stripe_width] = video[i][0][
            :, stripe_p :stripe_p + stripe_width]
        current = current + stripe_width

    return canvas

def mosaic_video(video_test,extended):
    middle = video_test.shape[1]
    video, canw, canh = align_frames(video_test[:,middle // 2: int(middle * (3/4))],video_test,extended)

    frames = []
    for stripe_p in range(10,video_test.shape[2] - 20,10): #range(10,video_test.shape[2] - 10,10) ###### do not forget to change 20 to 10

        mosaic_frame = creating_mosaic(video,stripe_p,canw,canh,middle)
        frames.append(mosaic_frame)

    return frames

def SIFT_algorithm(first,second):

    detector = cv2.SIFT_create()

    kp1, des1 = detector.detectAndCompute(first, None)
    kp2, des2 = detector.detectAndCompute(second, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M_affine, inliers = cv2.estimateAffinePartial2D(pts1, pts2)

    rotation_angle = 0

    if M_affine is not None:
        # Extract rotation angle from affine matrix
        rotation_angle = np.arctan2(M_affine[1, 0], M_affine[0, 0])

    return rotation_angle

def align_rotation(video):


    middle = video.shape[1]
    #[:,middle // 2: int(middle * (3/4))]
    for i in range(1,len(video)):
        # print(i)
        s_frame = np.array(Image.fromarray(video[i]).convert('L'))
        f_frame = np.array(Image.fromarray(video[i-1]).convert('L'))
        rotation = SIFT_algorithm(f_frame,s_frame)
        print(rotation)
        video[i] = mosaic_transformation(video[i],0,0,-rotation)

    return video


def boat_mosaic():
    video_test = media.read_video('Example inputs/my_video.mp4')[::-1]
    video_test = np.array(video_test)
    print(video_test.shape)

    # new_video = []
    # for frame in video_test:
    #     # Resize to more standard dimensions
    #     resized = cv2.resize(frame, (640, 480))  # Keep height, reduce width
    #     new_video.append(resized)
    #
    # video_test = np.array(new_video)

    ca = mosaic_video(video_test,0)
    ca = np.array(ca).astype(np.uint8)
    plt.imshow(ca[10])
    plt.show()
    # frames_resized = []
    # for frame in ca:
    #     # Resize to more standard dimensions
    #     resized = cv2.resize(frame, (1280, 208))  # Keep height, reduce width
    #     frames_resized.append(resized)

    # video_rgb= np.array(frames_resized)

    # reversed_video = video_rgb[::-1]
    # forward_backward = np.concatenate([video_rgb, reversed_video], axis=0)

    media.write_video('Example outputs/video new output.mp4', ca,fps=24)



#####################



def tree_mosaic():

    video_test = np.array(media.read_video('Example inputs/boat.mp4'))[:100]
    print(video_test.shape)

    video_test = align_rotation(video_test)

    ca = mosaic_video(video_test,0)
    ca = np.array(ca).astype(np.uint8)

    # for img in ca:
    #     plt.imshow(img)
    #     plt.show()

    frames_resized = []
    for frame in ca:
        # Resize to more standard dimensions
        resized = cv2.resize(frame, (300, 272))  # Keep height, reduce width
        frames_resized.append(resized)

    video_rgb = np.array(frames_resized)

    media.write_video('Example outputs/video_garden_output.mp4', video_rgb[::-1], fps=24)




tree_mosaic()
# boat_mosaic()





