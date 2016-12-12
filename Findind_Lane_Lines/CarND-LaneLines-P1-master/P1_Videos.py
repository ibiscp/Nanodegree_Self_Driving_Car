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

def find_lanes(lines, ysize):
    global last_left_angle, last_right_angle, last_mean_left, last_mean_right


    # print()
    # print(last_left_angle)
    # print(last_right_angle)

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

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = ((y2-y1)/(x2-x1))
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if angle > 0:
                interpolated_right.append([[(ysize-y2)/angle + x2, ysize, (inf-y2)/angle + x2, inf, length, angle]])
            elif angle < 0:
                interpolated_left.append([[(ysize-y2)/angle + x2, ysize, (inf-y2)/angle + x2, inf, length,angle]])
            else:
                print("Angle equal zero")

    # Calculate mean value for right side
    a = 0
    count = 0
    right_angle = 0
    for y in interpolated_right:
        for x1, y1, x2, y2, length, angle in y:
            if abs(angle/last_right_angle - 1) < 0.1 or last_right_angle == 0:
                mean_right = [mean_right[0]+x1*length, mean_right[1]+y1*length,
                              mean_right[2]+x2*length, mean_right[3]+y2*length]
                a += length
                count += 1
                right_angle += angle

    if a != 0 and sum(mean_left) < 10**6:
        mean_right = [[[int(mean_right[0]/a),int(mean_right[1]/a),int(mean_right[2]/a),int(mean_right[3]/a)]]]
        last_right_angle = right_angle / count
        last_mean_right = mean_right
    else:
        print("Division by zero")
        mean_right = last_mean_right

    # Calculate mean value for left side
    a = 0
    count = 0
    left_angle = 0
    for y in interpolated_left:
        for x1, y1, x2, y2, length, angle in y:
            if abs(angle/last_left_angle - 1) < 0.1 or last_left_angle == 0:
                mean_left = [mean_left[0]+x1*length,mean_left[1]+y1*length,
                              mean_left[2]+x2*length,mean_left[3]+y2*length]
                a += length
                count += 1
                left_angle += angle

    if a != 0 and sum(mean_left) < 10**50:
        mean_left = [[[int(mean_left[0]/a),int(mean_left[1]/a),int(mean_left[2]/a),int(mean_left[3]/a)]]]
        last_left_angle = left_angle/count
        last_mean_left = mean_left
    else:
        print("Division by zero")
        mean_left = last_mean_left

    return([mean_right[0], mean_left[0]])

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    #------------------------------Apply Color------------------------------#
    # Define color selection criteria
    red_threshold = 130
    green_threshold = 130
    blue_threshold = 0

    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)
    line_image = np.copy(image)

    # Define the vertices of a triangular mask
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex = [round(xsize / 2), round(ysize / 2)+40]

    # Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Perform a "bitwise or" to mask pixels below the threshold
    color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                       (image[:, :, 1] < rgb_threshold[1]) | \
                       (image[:, :, 2] < rgb_threshold[2])

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))

    # Mask color and region selection
    color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]

    #------------------------------Apply Canny------------------------------#
    # Convert image to gray
    gray_image = P1.grayscale(color_select)

    # Blur image
    blur_image = P1.gaussian_blur(gray_image, 3)

    # Define our parameters for Canny and run it
    high_threshold, _ = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_threshold = 0.5 * high_threshold

    # Apply Canny
    canny_image = P1.canny(blur_image, low_threshold, high_threshold)

    # ------------------------------Apply Hough------------------------------#
    # Create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(canny_image)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] / 2, imshape[0] / 2), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_image = P1.region_of_interest(canny_image, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # ------------------------------Extrapolate lines------------------------------#
    lanes = find_lanes(lines, ysize)

    hough_image = P1.hough_lines(masked_image,rho,theta,threshold,min_line_length,max_line_gap, lanes)

    draw_image = P1.weighted_img(hough_image, image)

    return draw_image

last_right_angle = 0
last_left_angle = 0

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip("solidYellowLeft.mp4")
yellow_clip = clip2.fl_image(process_image) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)