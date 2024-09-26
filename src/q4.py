import numpy as np
import cv2
from q4_aux import *

USE_PRE_SELECTED_POINTS = True # Set True to use the example points used in the rapport.
IMAGE_SELECTION = "Keble" # 'Amsterdam', 'Keble' or 'Paris'.

print("OpenCV Version: ", cv2.__version__)

# Image paths
PATH_IMG1, PATH_IMG2, PATH_IMG3 = get_image_paths(IMAGE_SELECTION)

# Load images
img1 = cv2.imread(PATH_IMG1)
img2 = cv2.imread(PATH_IMG2)
img3 = cv2.imread(PATH_IMG3)

(h1, w1, c1) = img1.shape

new_width = 400
new_height = int(new_width * h1 / w1)

img1 = cv2.resize(img1, (new_width, new_height))
img2 = cv2.resize(img2, (new_width, new_height))
img3 = cv2.resize(img3, (new_width, new_height))

# Variables to store points
points_selected = 0
X_init1 = []  # Points in image 1
X_final1 = []  # Points in image 2.1
X_final2 = []  # Points in image 2.2
X_init2 = []  # Points in image 3

# Clone images to allow resetting
clone_img1 = img1.copy()
clone_img2 = img2.copy()
clone_img3 = img3.copy()

if(USE_PRE_SELECTED_POINTS):
    X_init1, X_final1, X_init2, X_final2 = pre_selected_points(IMAGE_SELECTION)
else:

    # Function to select points on images
    def select_points(event, x, y, flags, param):
        global points_selected, X_init1, X_final1, X_final2, X_init2
        global img1, img2, img3

        if event == cv2.EVENT_LBUTTONDOWN:  # Click to select a point
            points_selected += 1
            if param == "Image1":
                X_init1.append([x, y])
                cv2.circle(img1, (x, y), 8, (0, 255, 255), 2)
                print(f"Point {points_selected} added to Image 1: ({x}, {y})")
            elif param == "Image2.1":
                X_final1.append([x, y])
                cv2.circle(img2, (x, y), 8, (0, 255, 255), 2)
                print(f"Point {points_selected} added to Image 2.1: ({x}, {y})")
            elif param == "Image2.2":
                X_final2.append([x, y])
                cv2.circle(img2, (x, y), 8, (0, 255, 255), 2)
                print(f"Point {points_selected} added to Image 2.2: ({x}, {y})")
            elif param == "Image3":
                X_init2.append([x, y])
                cv2.circle(img3, (x, y), 8, (0, 255, 255), 2)
                print(f"Point {points_selected} added to Image 3: ({x}, {y})")

    # Loops for visualization and point selection
    cv2.namedWindow("Image 1")
    cv2.setMouseCallback("Image 1", select_points, param="Image1")

    cv2.namedWindow("Image 2")
    cv2.setMouseCallback("Image 2", select_points, param="Image2.1")

    while True:
        cv2.imshow("Image 1", img1)
        cv2.imshow("Image 2", img2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") and points_selected >= 4:  # Requires at least 4 points
            points_selected = 0
            break

    cv2.setMouseCallback("Image 2", select_points, param="Image2.2")
    cv2.namedWindow("Image 3")
    cv2.setMouseCallback("Image 3", select_points, param="Image3")

    while True:
            cv2.imshow("Image 2", img2)
            cv2.imshow("Image 3", img3)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") and points_selected >= 4:  # Requires at least 4 points
                break

# Reset images
img1 = clone_img1
img2 = clone_img2
img3 = clone_img3

# Add boundary points for alignment
X_init1.append([max([point[0] for point in X_init1]), new_height - 1])
X_init1.append([max([point[0] for point in X_init1]), 0])
X_final2.append([min([point[0] for point in X_final2]), new_height - 1])
X_final2.append([min([point[0] for point in X_final2]), 0])
X_final1.append([max([point[0] for point in X_final1]), new_height - 1])
X_final1.append([max([point[0] for point in X_final1]), 0])
X_init2.append([min([point[0] for point in X_init2]), new_height - 1])
X_init2.append([min([point[0] for point in X_init2]), 0])

# Convert selected points to numpy arrays
X_init1 = np.asarray(X_init1, dtype=np.float32)
X_final1 = np.asarray(X_final1, dtype=np.float32)
X_final2 = np.asarray(X_final2, dtype=np.float32)
X_init2 = np.asarray(X_init2, dtype=np.float32)

# Apply color adjustments to images
img1 = fix_color(img1, img2, int(X_init1[0, 0]), int(X_init1[0, 1]), int(X_final1[0, 0]), int(X_final1[0, 1]))
img3 = fix_color(img3, img2, int(X_init2[0, 0]), int(X_init2[0, 1]), int(X_final2[0, 0]), int(X_final2[0, 1]))

# Estimate homographies
H1, status = cv2.findHomography(X_init1, X_final1)
H2, status = cv2.findHomography(X_init2, X_final2)

# Panorama settings
panorama_width = 1400
panorama_height = 500

# Calculate translation offsets
tx = panorama_width / 2 - new_width / 2
ty = panorama_height / 2 - new_height / 2

# Apply translations to homographies
M = np.float32([[1, 0, tx], [0, 1, ty]])
H1 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]) @ H1
H2 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]) @ H2

# Apply translation to the second image
img2 = cv2.warpAffine(img2, M, (panorama_width, panorama_height))

# Remove overlapping areas in images
img1[:, int(np.max(X_init1[:, 0])):] = [0, 0, 0]
img3[:, 0:int(np.min(X_init2[:, 0]))] = [0, 0, 0]

# Warp images according to homographies
img1_warp = cv2.warpPerspective(img1, H1, (panorama_width, panorama_height))
img3_warp = cv2.warpPerspective(img3, H2, (panorama_width, panorama_height))

# Define boundaries for blending between images
left_blend_start = tx  # Starting point of blending for the left image
right_blend_end = tx + new_width  # Starting point of blending for the right image
top_offset = ty  # Not used, but defines a vertical offset if needed
bottom_offset = ty + new_height  # Not used, similar to the top offset
blend_region_width = 40

# Iterate through each pixel in the panorama
for row in range(panorama_height):
    for col in range(panorama_width):
        # If the current pixel is almost black (indicating an unpopulated region)
        if np.all(img2[row, col] < 10):
            # Use img1_warp for the left half of the panorama
            if col < panorama_width / 2:
                img2[row, col] = img1_warp[row, col]
            # Use img3_warp for the right half
            else:
                img2[row, col] = img3_warp[row, col]

        # Blend img1 and img2 in the left overlap region
        elif left_blend_start <= col < left_blend_start + blend_region_width:
            blend_position = col - left_blend_start
            alpha = blend_position / blend_region_width
            img2[row, col] = (1 - alpha) * img1_warp[row, col] + alpha * img2[row, col]

        # Blend img2 and img3 in the right overlap region
        elif right_blend_end - blend_region_width <= col < right_blend_end:
            blend_position = col - (right_blend_end - blend_region_width)
            alpha = blend_position / blend_region_width
            img2[row, col] = (1 - alpha) * img2[row, col] + alpha * img3_warp[row, col]

# Display the final panorama
cv2.imshow("Panorama", img2)

k = cv2.waitKey(0)
if (k == ord("q")):
	cv2.destroyAllWindows()
elif (k == ord("s")):
	cv2.imwrite(f"img_rectified{IMAGE_SELECTION}.png",img2)
	cv2.destroyAllWindows()
