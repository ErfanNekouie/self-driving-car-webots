import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class KalmanFilter1D:
    def __init__(self, initial_state, initial_uncertainty, process_variance, measurement_variance):
        self.state_estimate = initial_state
        self.uncertainty = initial_uncertainty
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self):
        # Prediction step
        self.state_estimate = self.state_estimate  # In 1D, no change in state
        self.uncertainty = self.uncertainty + self.process_variance

    def update(self, measurement):
        # Update step
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
        self.state_estimate = self.state_estimate + kalman_gain * (measurement - self.state_estimate)
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

    def get_state(self):
        return self.state_estimate, self.uncertainty


def calculate_angle_between_vectors(u, v):
    # Ensure the vectors are numpy arrays
    u = np.array(u)
    v = np.array(v)

    # Calculate the dot product
    dot_product = np.dot(u.T, v)

    # Calculate the magnitudes (norms) of the vectors
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)

    # Clip cos_theta to avoid numerical issues (values should be in [-1, 1])
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calculate the angle in radians
    angle_rad = np.arccos(cos_theta)

    # Convert the angle to degrees (optional)
    angle_deg = np.degrees(angle_rad)

    return angle_rad, angle_deg


def warp(img, src_points, des_points):
    img_size = (320, 180)
    src = np.float32(src_points)
    dst = np.float32(des_points)
    transfer_matrices = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, transfer_matrices, img_size, flags=cv2.INTER_LINEAR)
    return warped


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


def line_indices(poly_fitted, non_zero_arr_x, non_zero_arr_y, margin):
    return ((non_zero_arr_x > (
            poly_fitted[0] * (non_zero_arr_y ** 2) + poly_fitted[1] * non_zero_arr_y + poly_fitted[2] - margin)) &
            (non_zero_arr_x < (poly_fitted[0] * (non_zero_arr_y ** 2) +
                               poly_fitted[1] * non_zero_arr_y + poly_fitted[2] + margin)))


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # plt.plot(histogram)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] // 2)
    left_side = histogram[:midpoint]
    right_side = histogram[midpoint:]
    left_x_base = np.argmax(left_side)
    right_x_base = np.argmax(right_side) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    n_windows = 8
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

    return left_x, lefty, right_x, righty


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

    return sum(curvature)


def fit_polynomial(binary_warped, plot_y):
    # Find our lane pixels first
    left_x, lefty, right_x, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial
    if len(left_x) > 0:
        left_fit = np.polyfit(lefty, left_x, 2)
    else:
        left_fit = None
    if len(right_x) > 0:
        right_fit = np.polyfit(righty, right_x, 2)
    else:
        right_fit = None

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

    return left_fit, right_fit, left_fitx, right_fitx


def draw_polynomial(image_inner, x_values, y_values, color=(0, 255, 0)):
    points = np.array([np.vstack((x_values, y_values)).astype(np.int32).T])

    # Draw the polynomial on the image
    for point in points[0]:
        cv2.circle(image_inner, tuple(point), 1, color, 12)


def fit_poly(img_shape, x_axis, y_axis, plot_y):
    polynomial_fit = np.polyfit(y_axis, x_axis, 2)
    plot_x = polynomial_fit[0] * plot_y ** 2 + polynomial_fit[1] * plot_y + polynomial_fit[2]

    return plot_x, polynomial_fit


def search_around_poly(binary_warped, poly_fit, plot_y):
    margin = 40

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
    right_fit_x, poly_fit_x = fit_poly(binary_warped.shape, right_x, right_y, plot_y)

    return right_fit_x, poly_fit_x


def one_line_dist_finder(x_fitted, half_road, indicator):
    if indicator == 'l':
        estimated_line_x = x_fitted + half_road
    elif indicator == 'r':
        estimated_line_x = x_fitted - half_road
    return estimated_line_x


# variables
source_points = [(350, 440), (0, 630), (1205, 630), (950, 440)]
desired_points = [(40, 0), (40, 180), (280, 180), (280, 0)]
center_points = (160, 180)
middle_line = center_points[0] * np.ones(180)

y_plot = np.linspace(0, 179, 180)
fit_left = []
fit_right = []
fitx_left = []
fitx_right = []
last_left_fitx = 0
last_right_fitx = 0
left_curve = 0
right_curve = 0
identifier = None

left_line_found = False
right_line_found = False

left_fitted_line = []
right_fitted_line = []

combiner = np.zeros((180, 320), dtype=np.uint8)
combiner[:, 30:300] = 1
combiner[:, 150:220] = 0

distance = 0

# taking input
# image = cv2.imread(r"./frames/frame-1003.jpg")


# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('./test1.avi')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

initial_state = 0.0
initial_uncertainty = 1.0
process_variance = 0.001
measurement_variance = 0.1

kf = KalmanFilter1D(initial_state, initial_uncertainty, process_variance, measurement_variance)

distances = []
new_measurements = []

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        # cv2.imshow('Frame', frame)

        # warping image and masking it
        warped_img = warp(frame, source_points, desired_points)
        warped_binary = binarize(warped_img)
        warped_binary = cv2.bitwise_and(combiner, warped_binary, mask=combiner).astype(np.float64)

        # finding the fitx and fity
        # if left_line_found and right_line_found:
        #     fitx_left, fit_left = search_around_poly(warped_binary, fit_left, y_plot)
        #     fitx_right, fit_right = search_around_poly(warped_binary, fit_right, y_plot)
        # else:
        fit_left, fit_right, fitx_left, fitx_right = fit_polynomial(warped_binary, y_plot)

        if fitx_left is not None:
            left_line_found = True
        if fitx_right is not None:
            right_line_found = True

        # finding the middle line
        if left_line_found and right_line_found:
            # comment the next line when running in simulator
            middle_line = (fitx_left + fitx_right) / 2
            # middle_line = (fitx_left[719] + fitx_right[719]) / 2
            left_curve = calculate_curvature(y_plot, fitx_left)
            right_curve = calculate_curvature(y_plot, fitx_right)
            # print(left_curve, right_curve)
            if abs(right_curve - left_curve) > 0.5:
                identifier = min(right_curve, left_curve)
        if (left_line_found and not right_line_found) or identifier == left_curve:
            middle_line = one_line_dist_finder(fitx_left, 130, 'l')
            # comment the next line when running in simulator
            fitx_right = one_line_dist_finder(middle_line, 130, 'l')
        if (right_line_found and not left_line_found) or identifier == right_curve:
            middle_line = one_line_dist_finder(fitx_right, 130, 'r')
            # comment the next line when running in simulator
            fitx_left = one_line_dist_finder(middle_line, 130, 'r')

        # finding distance
        new_distance = center_points[0] - middle_line[179]
        if abs(new_distance) > 40:
            distance = new_distance/10
        else:
            distance = new_distance
            
        distances.append(distance)

        # the following part is just for showing purposes
        # print('this is distance', distance)
        # print(middle_line.shape)
        # print((center_points[0]*np.ones(180)).shape)
        # print(calculate_angle_between_vectors(middle_line, center_points[0] * np.ones(180)))

        kf.predict()
        kf.update(distance)
        state_estimate, uncertainty = kf.get_state()
        new_measurements.append(state_estimate)
        # print(f"State estimate: {state_estimate}, Uncertainty: {uncertainty}")

        draw_polynomial(warped_img, middle_line, y_plot, (255, 0, 0))
        draw_polynomial(warped_img, fitx_right, y_plot, (0, 0, 255))
        draw_polynomial(warped_img, fitx_left, y_plot, (0, 0, 255))
        draw_polynomial(warped_img, center_points[0] * np.ones_like(y_plot), y_plot)

        # cv2.imshow('result', warped_img)
        # cv2.waitKey(1)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

plt.plot(distances)
plt.plot(new_measurements)
plt.show()

# plt.imshow(warped_img)
# plt.show()
