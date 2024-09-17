"""Homographe Python library for Vision 3D, CSC_5RO17_TA."""

import cv2
import numpy as np

print("Version d'OpenCV: ", cv2.__version__)

# Ouverture de l'image
PATH_IMG = './Images_Homographie/'

IMG = np.uint8(cv2.imread(PATH_IMG + 'Pompei.jpg'))

(h, w, c) = IMG.shape
print("Dimension de l'image :", h, 'lignes x', w, 'colonnes x', c, 'couleurs')


CLONE = IMG.copy()
POINTS_SELECTED = 0
X_INIT = []


def select_points(event, x, y, flags=None, param=None):
    """Record user selected points of an image."""
    # pylint: disable=unused-argument
    # pylint: disable=global-variable-not-assigned
    global POINTS_SELECTED, X_INIT
    global IMG, CLONE
    if event == cv2.EVENT_FLAG_LBUTTON:
        x_select, y_select = x, y
        POINTS_SELECTED += 1
        cv2.circle(IMG, (x_select, y_select), 8, (0, 255, 255), 1)
        cv2.line(IMG, (x_select - 8, y_select), (x_select + 8, y_select), (0, 255, 0), 1)
        cv2.line(IMG, (x_select, y_select - 8), (x_select, y_select + 8), (0, 255, 0), 1)
        X_INIT.append([x_select, y_select])
    elif event == cv2.EVENT_FLAG_RBUTTON:
        POINTS_SELECTED = 0
        IMG = CLONE.copy()


cv2.namedWindow('Image initiale')
cv2.setMouseCallback('Image initiale', select_points)


while True:
    cv2.imshow('Image initiale', IMG)
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q')) & (POINTS_SELECTED >= 4):
        break

# Conversion en array numpy
X_INIT = np.asarray(X_INIT, dtype=np.float32)
print('X_INIT =', X_INIT)
X_final = np.zeros((POINTS_SELECTED, 2), np.float32)
for i in range(POINTS_SELECTED):
    string_input = f'Correspondant de {X_INIT[i]} ? '
    X_final[i] = input(string_input).split(' ', 2)
print('X_final =', X_final)

# Votre code d'estimation de H ici

# Votre code d'estimation de H ici

# Juste un exemple pour afficher quelque chose
H = np.array([[1.1, 0.0, 10.0], [0.5, 0.9, -25.0], [0.0, 0.0, 1.0]])
img_warp = cv2.warpPerspective(CLONE, H, (w, h))
cv2.imshow('Image rectifiÃ©e', img_warp)
cv2.waitKey(0)
