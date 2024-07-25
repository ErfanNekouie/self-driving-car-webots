import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.use('TkAgg')

source_points = [(450, 370), (0, 600), (1180, 600), (880, 370)]
desired_points = [(200, 0), (200, 720), (1000, 720), (1000, 0)]
center_points = (640, 720)

left_line_found = False
right_line_found = False

left_fitted_line = []
right_fitted_line = []


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


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] // 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    n_windows = 9
    # Set the width of the windows +/- margin
    margin = 100
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

        # If you found > min_pix pixels, recenter next window on their mean position
        if len(good_left_indices) > min_pix:
            left_x_current = int(np.mean(non_zero_x[good_left_indices]))
        if len(good_right_indices) > min_pix:
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

    return left_x, lefty, right_x, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    left_x, lefty, right_x, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    print(left_x)
    print(lefty)

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

    # Plots the left and right polynomials on the lane lines
    if left_fitx is not None:
        plt.plot(left_fitx, plot_y, color='red')
    if right_fitx is not None:
        plt.plot(right_fitx, plot_y, color='red')

    return plot_y, left_fit, right_fit, left_fitx, right_fitx, out_img


def fit_poly(img_shape, left_x, lefty, right_x, righty):
    left_fit = np.polyfit(lefty, left_x, 2)
    right_fit = np.polyfit(righty, right_x, 2)
    plot_y = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    return left_fit, right_fit, plot_y, left_fitx, right_fitx


def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y +
                                    left_fit[2] - margin)) & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) +
                                                                           left_fit[1] * nonzero_y + left_fit[
                                                                               2] + margin)))
    right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y +
                                     right_fit[2] - margin)) & (nonzero_x < (right_fit[0] * (nonzero_y ** 2) +
                                                                             right_fit[1] * nonzero_y + right_fit[
                                                                                 2] + margin)))

    # Again, extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds]
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    # Fit new polynomials
    left_fit_, right_fit_, plot_y, left_fitx, right_fitx = fit_poly(binary_warped.shape, left_x, left_y, right_x,
                                                                    right_y)

    return plot_y, left_fit_, right_fit_, left_fitx, right_fitx


# startTime = datetime.now()
image = cv2.imread(r"./frames/frame-1586.jpg")

warped_img = warp(image, source_points, desired_points)
new_image = binarize(warped_img)
combiner = np.zeros_like(new_image, dtype=np.uint8)
combiner[:, 200:1060] = 1
new_image = cv2.bitwise_and(combiner, new_image, mask=combiner).astype(np.float64)

# gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
# hls = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HLS)

plot_y_, left_fit_, right_fit_, left_fitx_, right_fitx_, out_img_ = fit_polynomial(new_image)
# print(right_fitx)
# print(len(plot_y))
# print(f"right fit {right_fitx[719]}:{plot_y[719]}")
# print(f"left fit {left_fitx[719]}:{plot_y[719]}")
warped_img = warp(image, source_points, desired_points)
new_image = binarize(warped_img)
combiner = np.zeros_like(new_image, dtype=np.uint8)
combiner[:600, 100:1060] = 1
new_image = cv2.bitwise_and(combiner, new_image, mask=combiner).astype(np.float64)

plot_y_, left_fit_, right_fit_, left_fitx_, right_fitx_ = fit_polynomial(new_image)

if left_fitx_ is not None and right_fitx_ is not None:
    base_line = (right_fitx_ + left_fitx_) / 2

distance = center_points[0] - base_line[719]
print(f'distance: {distance}')

# print(
#     f"distance from left line {center_points[0] - left_fitx_[719]} and distance from right line"
#     f" {right_fitx_[719] - center_points[0]}")

# if left_line_found and right_line_found:


# new_image = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
# plt.imshow(warped_img)
# plt.show()
# h_channel = hls[:, :, 0]
# s_channel = hls[:, :, 1]
# v_channel = hls[:, :, 2]
#
# cv2.imshow("Image1", h_channel)
# cv2.waitKey(0)
# cv2.imshow("Image2", s_channel)
# cv2.waitKey(0)
# cv2.imshow("Image3", v_channel)
# cv2.waitKey(0)

# print(combiner)

# Perform the bitwise_and operation

# print(hello.shape)

# plot_image = cv2.cvtColor(hello, cv2.COLOR_BGR2RGB)

# plt.imshow(new_image)
# plt.show()
# print(datetime.now() - startTime)
# cv2.imshow("Image", new_image)
# cv2.waitKey(0)
