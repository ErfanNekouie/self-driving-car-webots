# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vehicle_driver_altino controller."""

from vehicle import Driver
from controller import Camera, Keyboard, LED
import time
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


# lane finding functions
def resize_image(input_image, width=None, height=None):
    # Load the image from the input path
    if input_image is None:
        print(f"Error: Could not load image from {input_path}")
        return

    # Get the original dimensions of the image
    original_height, original_width = input_image.shape[:2]

    # Calculate the new dimensions of the image
    if width is None:
        # Calculate the width from the given height while maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate the height from the given width while maintaining the aspect ratio
        aspect_ratio = original_height / original_width
        height = int(width * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_AREA)

    # Save the resized image to the output path

    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(1)


def warp(img, src_points, des_points):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    dst = np.float32(des_points)
    transfer_matrices = cv2.getPerspectiveTransform(src, dst)
    # un_transfer_matrices = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, transfer_matrices, img_size, flags=cv2.INTER_LINEAR)
    return warped  # un_transfer_matrices


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        _sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        _sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(_sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return s_binary


def mag_thresh(gray, sobel_kernel=3, thresh_mag=(0, 255)):
    x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.absolute(x_sobel)
    abs_sobel_y = np.absolute(y_sobel)
    magnitude = np.sqrt(np.square(abs_sobel_x) + np.square(abs_sobel_y))
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    s_binary = np.zeros_like(scaled_magnitude)
    s_binary[(scaled_magnitude >= thresh_mag[0]) & (scaled_magnitude <= thresh_mag[1])] = 1
    return s_binary


def dir_thresh(gray, sobel_kernel=3, thresh_=(0, np.pi / 2)):
    x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.absolute(x_sobel)
    abs_sobel_y = np.absolute(y_sobel)
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    s_binary = np.zeros_like(direction)
    s_binary[(direction >= thresh_[0]) & (direction <= thresh_[1])] = 1
    return s_binary


def hls_select(_image, thresh=(0, 255)):
    hls = cv2.cvtColor(_image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 1]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = abs_sobel_thresh(gray, 'x', 3, (20, 100))
    grad_y = abs_sobel_thresh(gray, 'y', 3, (20, 100))
    mag_binary = mag_thresh(gray, 9, (30, 100))
    dir_binary = dir_thresh(gray, 13, (0.7, 1.3))
    hls = hls_select(img, (180, 255))
    combined = np.zeros_like(dir_binary)
    combined[((grad_x == 1) | (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | hls == 1] = 1
    return combined.astype(np.uint8)


def draw_polynomial(image_inner, x_values, y_values, color=(0, 255, 0)):
    points = np.array([np.vstack((x_values, y_values)).astype(np.int32).T])

    # Draw the polynomial on the image
    for point in points[0]:
        cv2.circle(image_inner, tuple(point), 1, color, 12)


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    plt.plot(histogram)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] // 2)
    left_side = histogram[:midpoint]
    right_side = histogram[midpoint:]
    left_x_base = np.argmax(left_side)
    right_x_base = np.argmax(right_side) + midpoint

    # for idx in range(len(left_side)):
    #     if abs(left_side[idx] - left_x_base) > 5:
    #         left_side[idx] = 0
    # for idx in range(len(right_side)):
    #     if abs(right_side[idx] - right_x_base) > 5:
    #         right_side[idx] = 0

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    n_windows = 9
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    min_pix = 50

    # Set height of windows - based on n_windows above and image shape
    window_height = int(binary_warped.shape[0] // n_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    non_zero_y = np.array(nonzero[0])
    non_zero_x = np.array(nonzero[1])
    # Current positions to be updated later for each window in n_windows
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_indices = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                             (non_zero_x >= win_x_left_low) & (non_zero_x < win_x_left_high)).nonzero()[0]
        good_right_indices = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                              (non_zero_x >= win_x_right_low) & (non_zero_x < win_x_right_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        # print('right side', sum(good_right_indices - np.mean(good_right_indices)))
        # print('left side', sum(good_left_indices - np.mean(good_left_indices)))

        # If you found > min_pix pixels, recenter next window on their mean position
        # left_sum = 0
        # right_sum = 0
        if len(good_left_indices) > min_pix:
            # left_mean = np.mean(good_left_indices)
            # for idx in range(len(good_left_indices)):
            #     left_sum += (good_left_indices[idx] - left_mean) ** 2
            # left_sum /= len(good_left_indices)
            # print(left_sum)
            left_x_current = int(np.mean(non_zero_x[good_left_indices]))

        if len(good_right_indices) > min_pix:
            # right_mean = np.mean(good_right_indices)
            # for idx in range(len(good_right_indices)):
            #     right_sum += (good_right_indices[idx] - right_mean) ** 2
            # right_sum /= len(good_right_indices)
            # # print(right_sum)
            # if right_sum < 5000000:
            right_x_current = int(np.mean(non_zero_x[good_right_indices]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    left_x = non_zero_x[left_lane_indices]
    lefty = non_zero_y[left_lane_indices]
    right_x = non_zero_x[right_lane_indices]
    righty = non_zero_y[right_lane_indices]

    return left_x, lefty, right_x, righty


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    left_x, lefty, right_x, righty = find_lane_pixels(binary_warped)

    if len(left_x) > 0:
        left_fit = np.polyfit(lefty, left_x, 2)
    else:
        left_fit = None
    if len(right_x) > 0:
        right_fit = np.polyfit(righty, right_x, 2)
    else:
        right_fit = None
    # print("+++++++++++++++++++++++++++++++++++++++")
    # print(left_fit)
    # print(right_fit)

    # Generate x and y values for plotting
    plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line for left side!')
        left_fitx = None

    try:
        right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    except TypeError:
        right_fitx = None
        print('The function failed to fit a line for right side!')

    return plot_y, left_fit, right_fit, left_fitx, right_fitx


def fit_poly(img_shape, x_axis, y_axis):
    polynomial_fit = np.polyfit(y_axis, x_axis, 2)
    plot_y = np.linspace(0, img_shape[0] - 1, img_shape[0])
    plot_x = polynomial_fit[0] * plot_y ** 2 + polynomial_fit[1] * plot_y + polynomial_fit[2]

    return plot_x, plot_y, polynomial_fit


def calculate_curvature(x, y):
    # Ensure input arrays are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Calculate second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Calculate the curvature
    curvature = np.abs(ddx * dy - ddy * dx) / np.power(dx ** 2 + dy ** 2, 1.5)

    return curvature


def line_indices(poly_fitted, non_zero_arr_x, non_zero_arr_y, margin):
    return ((non_zero_arr_x > (
            poly_fitted[0] * (non_zero_arr_y ** 2) + poly_fitted[1] * non_zero_arr_y + poly_fitted[2] - margin)) &
            (non_zero_arr_x < (poly_fitted[0] * (non_zero_arr_y ** 2) +
                               poly_fitted[1] * non_zero_arr_y + poly_fitted[2] + margin)))


def search_around_poly(binary_warped, poly_fit):
    margin = 80

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # left_lane_indices = line_indices(left_fit, nonzero_x, nonzero_y, margin)
    right_lane_indices = line_indices(poly_fit, nonzero_x, nonzero_y, margin)

    # Again, extract left and right line pixel positions
    # left_x = nonzero_x[left_lane_indices]
    # left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    # Fit new polynomials
    # plot_y, left_fit_x, poly_fit_x = fit_poly(binary_warped.shape, left_x, left_y)
    plot_y, right_fit_x, poly_fit_x = fit_poly(binary_warped.shape, right_x, right_y)

    return plot_y, right_fit_x, poly_fit_x


# control functions

driver = Driver()
basicTimeStep = int(driver.getBasicTimeStep())

keyboard = Keyboard()
print("keyboard object called")
keyboard.enable(basicTimeStep)
print("keyboard enabled")

camera = Camera('jetcamera')
camera.enable(10)

sensorTimeStep = 4 * basicTimeStep
# speed refers to the speed in km/h at which we want Altino to travel
speed = 0
# angle refers to the angle (from straight ahead) that the wheels
# currently have
angle = 0

# This the Altino's maximum speed
# all Altino controllers should use this maximum value
maxSpeed = 2
minSpeed = -2
maxAngle = 0.65
minAngle = -0.65
# ensure 0 starting speed and wheel angle
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)
# defaults for this controller
right = False
printCounter = 0
i = 0

# Erfan variables
source_points = [(450, 370), (0, 600), (1180, 600), (880, 370)]
desired_points = [(200, 0), (200, 720), (1000, 720), (1000, 0)]
center_points = (640, 720)

left_line_found = False
right_line_found = False

left_fitted_line = []
right_fitted_line = []

base_line = center_points[0] * np.ones(720)
# Erfan variables

turn = True
n = 0
last_left_fitx = 0
last_right_fitx = 0
left_curve = 0
right_curve = 0

# Mohammad variables
e = 0
sum1 = 0
kp = 0.02
ki = 0
kd = 0
# Mohammad variables

# frame_num = 0
while driver.step() != -1:
    # print("speed:",speed)
    # print("Hello World!")
    # if right :
    # i += 1
    # else:
    # i -= 1

    # if i > 60 :
    # right = False
    # elif i < -60:
    # right = True
    # if(i < 0):
    # driver.setSteeringAngle(0.3)
    # else:
    # driver.setSteeringAngle(-0.3)
    '''
    key = keyboard.getKey()

    if (key == ord('W')):
        if (speed < maxSpeed):
            speed = speed + 0.01

    elif (key == ord('S')):
        if (speed > minSpeed):
            speed = speed - 0.01
    # else:
    # if(speed > 0):
    # speed -= 0.01
    # else:
    # speed += 0.01
    if (key == ord('Q')):
        speed = 0

    if (key == ord('W') + ord('D') or key == ord('D')):
        if (angle < maxAngle):
            angle = angle + 0.05

    elif (key == ord('S') + ord('A') or key == ord('A')):
        if (angle > minAngle):
            angle = angle - 0.05

    else:
        if (angle > 0.1):
            angle -= 0.1
        elif (angle < -0.1):
            angle += 0.1
        else:
            angle = 0
            '''
    n += 1
    img = camera.getImage()
    image = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    frame = image[:, :, :3]

    ############################
    """lane detection phase"""
    # startTime = datetime.now()
    # image = cv2.imread(r"./frames/frame-1586.jpg")

    warped_img, reverse_mat = warp(frame, source_points, desired_points)
    new_image = binarize(warped_img)
    combiner = np.zeros_like(new_image, dtype=np.uint8)
    combiner[:, 200:1060] = 1
    combiner[:, 400:850] = 0
    # for col in range(0, 400, 10):
    #     for row in range(0, 400, 10):
    #         combiner[row - 10:row, 450 + row:] = 0
    new_image = cv2.bitwise_and(combiner, new_image, mask=combiner).astype(np.float64)

    # gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    # hls = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HLS)

    plot_y_, left_fit_, right_fit_, left_fitx_, right_fitx_ = fit_polynomial(new_image)
    # print(right_fitx)
    # print(len(plot_y))
    # print(f"right fit {right_fitx[719]}:{plot_y[719]}")
    # print(f"left fit {left_fitx[719]}:{plot_y[719]}")

    # cv2.imshow("Image", new_image)
    # cv2.waitKey(0)
    base_line = center_points[0] * np.ones(720)

    if left_fitx_ is not None:
        left_curve = sum(calculate_curvature(plot_y_, left_fitx_))
        # last_left_fitx = left_fitx_
    else:
        plot_y_, last_left_fitx, left_fit_ = search_around_poly(warped_img, left_fit_)

    if right_fitx_ is not None:
        right_curve = sum(calculate_curvature(plot_y_, right_fitx_))
        # last_right_fitx = right_fitx_
    else:
        plot_y_, last_right_fitx, right_fit_ = search_around_poly(warped_img, right_fit_)

    if abs(left_curve - right_curve) > 0.5 and (right_fitx_ is not None and left_fitx_ is not None):
        if left_curve > right_curve:
            last_right_fitx = right_fitx_
        else:
            last_left_fitx = left_fitx_
    else:
        last_left_fitx = left_fitx_
        last_right_fitx = right_fitx_

    base_line = (last_right_fitx + last_left_fitx) / 2

    distance = base_line[719] - center_points[0]

    # draw_polynomial(warped_img, base_line, plot_y_, (255, 0, 0))
    # draw_polynomial(warped_img, right_fitx_, plot_y_, (0, 0, 255))
    # draw_polynomial(warped_img, left_fitx_, plot_y_, (0, 0, 255))
    # draw_polynomial(warped_img, center_points[0] * np.ones_like(plot_y_), plot_y_)

    # Show the image
    # resized_image = cv2.resize(warped_img, (400, 300), interpolation=cv2.INTER_AREA)
    # cv2.imshow("Quadratic Polynomial", resized_image)
    # cv2.waitKey(1)

    # resize_image(frame, width=400)
    ###########################################  #
    # control code
    if (-200 < distance and distance < 200):
        derror = distance - e
        e = distance
        sum1 += distance
        angle = kp * distance + ki * sum1 + kd * derror
        speed = 0.5
    else:
        print("else distance")
        speed = -0.5
        if (e < 0):
            angle = 0.30
            cv2.waitKey(500)
        else:
            angle = -0.30
            cv2.waitKey(500)
    driver.setSteeringAngle(angle)
    driver.setCruisingSpeed(speed)
writer.release()
cv2.destroyAllWindows()
