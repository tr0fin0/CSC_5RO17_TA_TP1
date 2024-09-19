"""Homographe Python library for Vision 3D, CSC_5RO17_TA."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_homograph(src_pts: np.array, dst_pts: np.array, method: str = 'DLT') -> np.array:
    """
    Return the homograph matrix between source points in image and destination points.

    Note: Direct Linear Transformation (DLT) method with Singular Values Decomposition (SVD) used.

    Args:
        src_pts (np.array) : reference points from source image.
        dst_pts (np.array) : reference points from destination image.
        method (str, optional) : algorithm method used. Default is 'DLT'.
    """
    # pylint: disable=too-many-locals
    func_name = get_homograph.__name__

    if src_pts.shape != dst_pts.shape:
        print(f'{func_name}: invalid input. {src_pts.shape} != {dst_pts.shape}')
        return None

    match method.upper():
        case 'DLT':
            # Build DLT matrix A.
            dlt = []
            for src_point, dst_point in zip(src_pts, dst_pts):
                x_1, y_1 = src_point
                x_2, y_2 = dst_point

                dlt.append([-x_1, -y_1, -1, 0, 0, 0, x_2 * x_1, x_2 * y_1, x_2])  # a_x line
                dlt.append([0, 0, 0, -x_1, -y_1, -1, y_2 * x_1, y_2 * y_1, y_2])  # a_y line

            # Compute homograph matrix H via SVD.
            _, _, eigen_values = np.linalg.svd(dlt)
            h11, h12, h13, h21, h22, h23, h31, h32, h33 = eigen_values[-1]  # smallest eigen values.

            homograph = [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]

            return np.asarray(homograph, dtype=np.float32)

        case 'CV2':
            homograph, _ = cv2.findHomography(src_pts, dst_pts)

            return homograph

        case _:
            print(f'{func_name}: invalid method. {method} not defined, returning None.')

            return None


def get_normalization(image: np.array) -> np.array:
    """
    Return normalization matrix T of image array.

    Note: image normalized to a 2 x 2 square with origin in its center.

    Args:
        image (np.array) : array representation of an image.
    """
    h, w, _ = image.shape  # height and width of image

    return np.asarray(
        [
            [2 / w, 0, -1],
            [0, 2 / h, -1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def normalize_points(points: np.array, norm: np.array) -> np.array:
    """
    Return points normalized by matrix norm.

    Args:
        points (np.array) : array of points [x, y] coordenates.
        norm (np.array) : normalization matrix.
    """
    points_normalized = []

    for point in points:
        x, y = point
        points_normalized.append((norm @ np.transpose([x, y, 1]))[:2])

    return np.array(points_normalized, dtype=np.float32)


def get_reprojection_error(homograph: np.array, src_pts: np.array, dst_pts: np.array) -> float:
    """
    Return reprojection error of homograph.

    Note: how closely the projected points via the homograph match the actual points in the image.

    Args:
        homograph (np.array) : homograph matrix.
        src_pts (np.array) : source image points.
        dst_pts (np.array) : destination image points.
    """
    projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), homograph).reshape(-1, 2)
    error = np.sqrt(np.sum((projected_pts - dst_pts) ** 2, axis=1))

    return np.mean(error)


def q3() -> None:
    """Compute question 3 answer."""
    # pylint: disable=too-many-locals
    func_name = q3.__name__

    # Function initialition
    print(f'{func_name}: init')
    print(f'OpenCV: v{cv2.__version__}')

    file_name = 'Pompei.jpg'
    path_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'Images_Homographie', file_name)
    )

    # get image source
    src_img = cv2.cvtColor(cv2.imread(path_file), cv2.COLOR_BGR2RGB)

    # define reference points
    src_pts = np.asarray(
        [
            [120, 23],
            [73, 287],
            [409, 299],
            [386, 22],
        ],
        dtype=np.float32,
    )
    dst_pts = np.asarray(
        [
            [83, 0],
            [83, 333],
            [416, 333],
            [416, 0],
        ],
        dtype=np.float32,
    )

    # get image corrected, method CV2
    homograph_0 = get_homograph(src_pts, dst_pts, 'CV2')
    dst_img_0 = cv2.warpPerspective(src_img, homograph_0, (src_img.shape[1], src_img.shape[0]))

    # get image corrected, method DLT
    homograph_1 = get_homograph(src_pts, dst_pts)
    dst_img_1 = cv2.warpPerspective(src_img, homograph_1, (src_img.shape[1], src_img.shape[0]))

    # get image corrected, method DLT normalized
    norm = get_normalization(src_img)
    src_pts_norm = normalize_points(src_pts, norm)
    dst_pts_norm = normalize_points(dst_pts, norm)

    homograph_norm = get_homograph(src_pts_norm, dst_pts_norm)
    homograph_2 = np.linalg.inv(norm) @ homograph_norm @ norm  # @ matrix multiplication operator

    dst_img_2 = cv2.warpPerspective(src_img, homograph_2, (src_img.shape[1], src_img.shape[0]))

    # plot images side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), num='Homograph Analysis')
    fig.suptitle('Homograph Analysis')

    axs[0].imshow(dst_img_0)
    axs[0].set_title(
        f'[{get_reprojection_error(homograph_0, src_pts, dst_pts):.2e}] CV2 method, benchmark'
    )

    axs[1].imshow(dst_img_1)
    axs[1].set_title(f'[{get_reprojection_error(homograph_1, src_pts, dst_pts):.2e}] DLT method')

    axs[2].imshow(dst_img_2)
    axs[2].set_title(
        f'[{get_reprojection_error(homograph_2, src_pts, dst_pts):.2e}] DLT method normalized'
    )

    image_name = 'CSC_5RO17_TP1_Q3.png'
    images_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'output', image_name)
    )

    plt.tight_layout()
    plt.savefig(f'{images_folder}')
    plt.show()
