"""Homographe Python library for Vision 3D, CSC_5RO17_TA."""

import os

import cv2
import numpy as np


def q2() -> None:
    """Question 2 algorithm."""
    # package versions
    print(f'OpenCV: {cv2.__version__}')

    # get image source
    file_name = 'Pompei.jpg'
    path_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'Images_Homographie', file_name)
    )

    image_source = cv2.imread(path_file)
    cv2.imshow('image_source', image_source)
    print(f'image_source: {image_source.shape}')

    # define reference points
    points_source = np.asarray(
        [
            [120, 23],
            [73, 287],
            [409, 299],
            [386, 22],
        ],
        dtype=np.float32,
    )
    points_corrected = np.asarray(
        [
            [83, 0],
            [83, 333],
            [416, 333],
            [416, 0],
        ],
        dtype=np.float32,
    )

    # calculate homography
    H, _ = cv2.findHomography(points_source, points_corrected)
    print(f'H:\n{H}')

    # get corrected image
    image_corrected = cv2.warpPerspective(
        image_source, H, (image_source.shape[1], image_source.shape[0])
    )
    print(f'image_corrected: {image_corrected.shape}')
    cv2.imshow('image_corrected', image_corrected)
    cv2.waitKey(0)
