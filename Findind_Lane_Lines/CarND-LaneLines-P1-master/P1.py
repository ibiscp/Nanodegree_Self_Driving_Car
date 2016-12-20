#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_poly(img, lines, color_lane=[255, 0, 0], color_road=[0,255,0,50]):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            pts = np.array([[x1-10,y1],[x1+10,y1], [x2+2,y2],[x2-2,y2]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img,[pts], color_lane)

    # Paint road in green
    # pts = np.array([[lines[1][0][0]+10,y1], [lines[1][0][2]+2,y2], [lines[0][0][2]-2,y2], [lines[0][0][0]-10,y1]], np.int32)
    # pts = pts.reshape((-1,1,2))
    # cv2.fillPoly(img, [pts], color_road)

    # Draw midle line
    # cv2.line(img, (int((lines[0][0][0] + lines[1][0][0])/2), y1), (int((lines[0][0][2] + lines[1][0][2])/2), y2), [255,255,255], thickness=1)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, lines, figure):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    #lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                        maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    if figure == "polygon":
        draw_poly(line_img, lines)
    else:
        draw_lines(line_img, lines)

    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def find_lanes(lines, ysize, inf):
    """
    Function that extrapolate and calculates the lines returned by the Hough Transformation

    :param lines: the output from the Hough Transformation
    :return: two lines, one for the right and one for the left lane
    """
    interpolated_right = []
    interpolated_left = []
    mean_right = [0,0,0,0]
    mean_left = [0,0,0,0]

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = ((y2-y1)/(x2-x1))
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Discard angles too discrepant
            if 1 > angle > 0.5:
                interpolated_right.append([[(ysize-y2)/angle + x2, ysize, (inf-y2)/angle + x2, inf, length, angle]])
            elif -1 < angle < -0.5:
                interpolated_left.append([[(ysize-y2)/angle + x2, ysize, (inf-y2)/angle + x2, inf, length, angle]])

    # Calculate mean value for right side
    a = 0
    for y in interpolated_right:
        for x1, y1, x2, y2, length, angle in y:
            mean_right = [mean_right[0]+x1*length, mean_right[1]+y1*length,
                          mean_right[2]+x2*length, mean_right[3]+y2*length]
            a += length

    if a != 0:
        mean_right = [[[int(mean_right[0]/a),int(mean_right[1]/a),int(mean_right[2]/a),int(mean_right[3]/a)]]]
    else:
        # print("Division by zero")
        mean_right = [[[0, 0, 0, 0]]]

    # Calculate mean value for left side
    a = 0
    for y in interpolated_left:
        for x1, y1, x2, y2, length, angle in y:
            mean_left = [mean_left[0]+x1*length,mean_left[1]+y1*length,
                          mean_left[2]+x2*length,mean_left[3]+y2*length]
            a += length

    if a != 0:
        mean_left = [[[int(mean_left[0]/a),int(mean_left[1]/a),int(mean_left[2]/a),int(mean_left[3]/a)]]]
    else:
        # print("Division by zero")
        mean_left = [[[0, 0, 0, 0]]]

    return([mean_right[0], mean_left[0]])

def save_image(data, fn):

    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi = height)
    plt.close()

def batch_analyze(path, road):

    images = filter( lambda f: not f.startswith(('Process', 'Final')), os.listdir(path))

    for i in images:
        # Open image
        image = mpimg.imread(path + i)
        print(i)

        # create a CLAHE object (Arguments are optional).
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)

        #------------------------------Apply Color------------------------------#
        # Grab the x and y size and make a copy of the image
        ysize = image.shape[0]
        xsize = image.shape[1]
        color_select = np.copy(image)

        if (road is "curve"):
            # Yellow threshold
            yellow_start = [200, 150, 0]
            yellow_end = [260, 210, 150]

            # White threshold
            white_start = [210, 190, 180]
            white_end = [260, 260, 250]

            yellow_threshold =  (image[:, :, 0] < yellow_start[0]) | (image[:, :, 0] > yellow_end[0]) | \
                                (image[:, :, 1] < yellow_start[1]) | (image[:, :, 1] > yellow_end[1]) | \
                                (image[:, :, 2] < yellow_start[2]) | (image[:, :, 2] > yellow_end[2])

            white_threshold =  (image[:, :, 0] < white_start[0]) | (image[:, :, 0] > white_end[0]) | \
                                (image[:, :, 1] < white_start[1]) | (image[:, :, 1] > white_end[1]) | \
                                (image[:, :, 2] < white_start[2]) | (image[:, :, 2] > white_end[2])

            color_thresholds = yellow_threshold & white_threshold
        else:
            # Define color selection criteria
            rgb_threshold = [120, 150, 50]

            # Perform a "bitwise or" to mask pixels below the threshold
            color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                               (image[:, :, 1] < rgb_threshold[1]) | \
                               (image[:, :, 2] < rgb_threshold[2])

        # Mask color and region selection
        color_select[color_thresholds] = [0, 0, 0]

        # This time we are defining a four sided polygon to mask
        imshape = image.shape
        inf = 450
        vertices = np.array([[(0, imshape[0]), (imshape[1] / 2 - 50, inf), (imshape[1] / 2 + 80, inf), (imshape[1], imshape[0])]], dtype=np.int32)
        masked_image = region_of_interest(color_select, vertices)

        #------------------------------Apply Canny------------------------------#
        # Convert image to gray
        gray_image = grayscale(masked_image)

        # Blur image
        blur_image = gaussian_blur(gray_image, 3)

        # Define our parameters for Canny and run it
        high_threshold, _ = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_threshold = 0.5 * high_threshold

        # Apply Canny
        canny_image = canny(blur_image, low_threshold, high_threshold)

        # ------------------------------Apply Hough------------------------------#
        # Define the Hough transform parameters
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        lines = cv2.HoughLinesP(canny_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        # Draw the lines on the edge image
        lines_edges = hough_lines(canny_image,rho,theta,threshold,min_line_length,max_line_gap, lines, "line")

        # ------------------------------Extrapolate lines------------------------------#
        lanes = find_lanes(lines, ysize, inf)

        hough_image = hough_lines(canny_image,rho,theta,threshold,min_line_length,max_line_gap, lanes, "polygon")

        draw_image = weighted_img(hough_image, image)

        # ------------------------------Plot and save images------------------------------#

        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(232)
        plt.title('Color Selection')
        plt.imshow(masked_image)
        plt.axis('off')
        plt.subplot(233)
        plt.title('Canny Edges')
        plt.imshow(canny_image, cmap='Greys_r')
        plt.axis('off')
        plt.subplot(234)
        plt.title('Hough Lines')
        plt.imshow(lines_edges)
        plt.axis('off')
        plt.subplot(235)
        plt.title('Hough Transformation')
        plt.imshow(hough_image)
        plt.axis('off')
        plt.subplot(236)
        plt.title('Final Image')
        plt.imshow(draw_image)
        plt.axis('off')

        plt.savefig(path + 'Process ' + i)

        save_image(draw_image, path + 'Final ' + i)
        save_image(masked_image, path + 'Process Color ' + i)

        #plt.show()

road = "curve"
road = "straight"
# batch_analyze("test_images/")
# batch_analyze("video_images/")