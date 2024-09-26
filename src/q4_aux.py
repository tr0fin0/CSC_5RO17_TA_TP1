import numpy as np
import cv2

def fix_color(img_source, img_reference, x_source, y_source, x_ref, y_ref):

    print(img_source.shape)

    # Get pixel colors for color adjustment
    color_source = img_source[y_source, x_source].astype(np.float32)
    color_reference = img_reference[y_ref, x_ref].astype(np.float32)

    # Calculate the necessary color adjustments
    diff = color_reference - color_source

    # Apply color adjustments to image
    img_source = cv2.add(img_source.astype(np.float32), diff)

    # Clip pixel values to valid range
    img_source = np.clip(img_source, 0, 255).astype(np.uint8)

    return img_source

def get_image_paths(image_title):

    if(image_title == "Amsterdam"):
        PATH_IMG1 = '../Images_Homographie/Amst-1.jpg'
        PATH_IMG2 = '../Images_Homographie/Amst-2.jpg'
        PATH_IMG3 = '../Images_Homographie/Amst-3.jpg'

    if(image_title == "Keble"):
        PATH_IMG1 = '../Images_Homographie/keble_a.jpg'
        PATH_IMG2 = '../Images_Homographie/keble_b.jpg'
        PATH_IMG3 = '../Images_Homographie/keble_c.jpg'

    if(image_title == "Paris"):
        PATH_IMG1 = '../Images_Homographie/paris_a.jpg'
        PATH_IMG2 = '../Images_Homographie/paris_b.jpg'
        PATH_IMG3 = '../Images_Homographie/paris_c.jpg'

    return PATH_IMG1, PATH_IMG2, PATH_IMG3

def pre_selected_points(image_title):

    if(image_title == "Amsterdam"):
        X_init1 = [[389, 100], [385, 189], [372, 198], [359, 127], [393, 131], [346, 161]]
        X_final1 = [[42, 103], [36, 190], [23, 199], [11, 126], [46, 134], [0, 165]]
        X_final2 = [[369, 75], [327, 97], [380, 180], [340, 176], [323, 137], [377, 103]]
        X_init2 = [[67, 76], [22, 95], [75, 179], [37, 175], [18, 136], [75, 105]]

    if(image_title == "Keble"):
        X_init1 = [[357, 16], [377, 200], [272, 206], [178, 165], [178, 76], [391, 87]]
        X_final1 = [[199, 25], [214, 201], [111, 206], [12, 164], [14, 68], [228, 94]]
        X_final2 = [[320, 24], [365, 157], [193, 201], [177, 93], [271, 68]]
        X_init2 = [[153, 30], [194, 161], [18, 203], [1, 88], [103, 69]]

    if(image_title == "Paris"):
        X_init1 = [[205, 122], [318, 183], [223, 174], [198, 186]]
        X_final1 = [[27, 117], [145, 183], [44, 175], [13, 186]]
        X_final2 = [[389, 137], [358, 218], [233, 151], [390, 160]]
        X_init2 = [[246, 131], [219, 204], [105, 139], [246, 154]]

    return X_init1, X_final1, X_init2, X_final2




