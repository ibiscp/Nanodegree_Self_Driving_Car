from random import randint

__author__ = 'Ibis'

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import P1
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

def find_lanes(lines, ysize, image):
    global last_left_angle, last_right_angle, last_mean_left, last_mean_right
    """
    Function that extrapolate and calculates the lines returned by the Hough Transformation

    :param lines: the output from the Hough Transformation
    :return: two lines, one for the right and one for the left lane
    """
    interpolated_right = []
    interpolated_left = []
    mean_right = [0,0,0,0]
    mean_left = [0,0,0,0]
    inf = 330
    angle_dif = 0.2

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = ((y2-y1)/(x2-x1))
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if angle > 0.5:
                interpolated_right.append([[(ysize-y2)/angle + x2, ysize, (inf-y2)/angle + x2, inf, length, angle]])
            elif angle < -0.5:
                interpolated_left.append([[(ysize-y2)/angle + x2, ysize, (inf-y2)/angle + x2, inf, length,angle]])
            # else:
            #     print("Angle equal zero")
            #     P1.save_image(image, 'video_images/' + str(randint(10000,99999)) + '.jpeg')

    # Calculate mean value for right side
    a = 0
    count = 0
    right_angle = 0
    for y in interpolated_right:
        for x1, y1, x2, y2, length, angle in y:
            if abs(angle/last_right_angle - 1) < angle_dif or last_right_angle == 0:
                mean_right = [mean_right[0]+x1*length, mean_right[1]+y1*length,
                              mean_right[2]+x2*length, mean_right[3]+y2*length]
                a += length
                count += 1
                right_angle += angle

    if a != 0:
        mean_right = [[[int(mean_right[0]/a),int(mean_right[1]/a),int(mean_right[2]/a),int(mean_right[3]/a)]]]
        last_right_angle = right_angle / count
        last_mean_right = mean_right
    else:
        P1.save_image(image, 'video_images/' + str(randint(10000,99999)) + '.jpeg')
        # print("Division by zero")
        mean_right = last_mean_right

    # Calculate mean value for left side
    a = 0
    count = 0
    left_angle = 0
    for y in interpolated_left:
        for x1, y1, x2, y2, length, angle in y:
            if abs(angle/last_left_angle - 1) < angle_dif or last_left_angle == 0:
                mean_left = [mean_left[0]+x1*length,mean_left[1]+y1*length,
                              mean_left[2]+x2*length,mean_left[3]+y2*length]
                a += length
                count += 1
                left_angle += angle

    if a != 0:
        mean_left = [[[int(mean_left[0]/a),int(mean_left[1]/a),int(mean_left[2]/a),int(mean_left[3]/a)]]]
        last_left_angle = left_angle/count
        last_mean_left = mean_left
    else:
        P1.save_image(image, 'video_images/' + str(randint(10000,99999)) + '.jpeg')
        # print("Division by zero")
        mean_left = last_mean_left

    return([mean_right[0], mean_left[0]])

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    #------------------------------Apply Color------------------------------#
    # Define color selection criteria
    red_threshold = 120
    green_threshold = 150
    blue_threshold = 50
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)

    # Perform a "bitwise or" to mask pixels below the threshold
    color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                       (image[:, :, 1] < rgb_threshold[1]) | \
                       (image[:, :, 2] < rgb_threshold[2])

    # Mask color and region selection
    color_select[color_thresholds] = [0, 0, 0]

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] / 2 - 50, 330), (imshape[1] / 2 + 50, 330), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_image = P1.region_of_interest(color_select, vertices)

    #------------------------------Apply Canny------------------------------#
    # Convert image to gray
    gray_image = P1.grayscale(masked_image)

    # Blur image
    blur_image = P1.gaussian_blur(gray_image, 3)

    # Define our parameters for Canny and run it
    high_threshold, _ = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5 * high_threshold

    # Apply Canny
    canny_image = P1.canny(blur_image, low_threshold, high_threshold)

    # ------------------------------Apply Hough------------------------------#
    # Define the Hough transform parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(canny_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # ------------------------------Extrapolate lines------------------------------#
    lanes = find_lanes(lines, ysize, image)

    hough_image = P1.hough_lines(canny_image,rho,theta,threshold,min_line_length,max_line_gap, lanes, 'polygon')

    draw_image = P1.weighted_img(hough_image, image)

    return draw_image

last_right_angle = 0
last_left_angle = 0

# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
#
# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip("solidYellowLeft.mp4")
# yellow_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!
# yellow_clip.write_videofile(yellow_output, audio=False)

curve_output = 'curve.mp4'
clip3 = VideoFileClip("challenge.mp4")
curve_clip = clip3.fl_image(process_image) #NOTE: this function expects color images!!
curve_clip.write_videofile(curve_output, audio=False)