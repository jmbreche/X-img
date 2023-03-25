"""
A module for processing images using OpenCV.

Functions:
- window_img: Applies windowing and rescaling to DICOM image.
- grayscale_img: Converts an image to grayscale.
- otsu_img: Applies Otsu's thresholding to image.
- edge_img: Applies edge detection to image using the Canny algorithm.
- equalize_img: Applies contrast-limited adaptive hist ogram equalization to enhance image contrast.

Classes:
- ProcImg: A collection of images with methods for batch processing.
"""


import os
import cv2
import numpy as np
import pydicom as dcm
import matplotlib.pyplot as plt


__all__ = ["ProcImg", "window_img", "grayscale_img", "otsu_img", "edge_img", "equalize_img"]


class ProcImg:
    def __init__(self, paths):
        """
        Create a ProcImg object from a list of image file paths.

        Args:
            paths (iterable): A list or other iterable containing the file paths of the images to load.
        """

        self.imgs = []

        for path in paths:
            if os.path.splitext(path)[1] in [".jpg"]:
                self.imgs.append(cv2.imread(os.path.normpath(str(path))))
            elif os.path.splitext(path)[1] in [".dcm"]:
                self.imgs.append(dcm.dcmread(os.path.normpath(str(path))))

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to all images.

        Args:
            func (callable): The function to apply to each image in the collection. Must take an image as the first
                             input parameter, followed by any additional input parameters specified using *args and
                             **kwargs.
            *args: Any additional input parameters to pass to the function.
            **kwargs: Any additional keyword input parameters to pass to the function.

        Returns:
            None
        """

        self.imgs = [func(img, *args, **kwargs) for img in self.imgs]

    def show_all(self):
        """
        Display all images.

        Returns:
            None
        """

        for i in range(len(self.imgs)):
            self.show(i)

    def show(self, idx):
        """
        Display image at specified index.

        Args:
            idx (int): Index of image.

        Returns:
            None
        """

        if isinstance(self.imgs[idx], np.ndarray):
            plt.imshow(self.imgs[idx], cmap="gray")
            plt.show()
        elif isinstance(self.imgs[idx], dcm.dataset.FileDataset):
            plt.imshow(self.imgs[idx].pixel_array, cmap="gray")
            plt.show()


def window_img(data, level=None, width=None):
    """
    Applies windowing and rescaling to DICOM image.

    Args:
        data (FileDataset): A DICOM file containing the image data.
        level (float): The window level value to use. If None, the value is obtained from the DICOM file metadata.
        width (float): The window width value to use. If None, the value is obtained from the DICOM file metadata.

    Returns:
        ndarray: The windowed and rescaled image as a NumPy array.
    """

    intercept = data[('0028', '1052')].value
    slope = data[('0028', '1053')].value

    intercept = int(intercept[0] if isinstance(intercept, dcm.multival.MultiValue) else intercept)
    slope = int(slope[0] if isinstance(slope, dcm.multival.MultiValue) else slope)

    img = data.pixel_array * slope + intercept

    level = data[('0028', '1050')].value if level is None else level * (np.max(img) - np.min(img)) + np.min(img)
    width = data[('0028', '1051')].value if width is None else width * (np.max(img) - np.min(img))

    level = int(level[0] if isinstance(level, dcm.multival.MultiValue) else level)
    width = int(width[0] if isinstance(width, dcm.multival.MultiValue) else width)

    img_min = level - width // 2
    img_max = level + width // 2

    img[img < img_min] = img_min
    img[img > img_max] = img_max

    return img


def grayscale_img(img):
    """
    Converts an image to grayscale.

    Args:
        img (ndarray): Input image.

    Returns:
        ndarray: Grayscaled image.
    """

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsu_img(img):
    """
    Applies Otsu's thresholding to image.

    Args:
        img (ndarray): Input image.

    Returns:
        ndarray: Image obtained by applying Otsu's thresholding.
    """

    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def edge_img(img, sigma=.1, lower=None, upper=None):
    """
    Applies edge detection to image using the Canny algorithm.

    Args:
        img (ndarray): Input image.
        sigma (float): Controls the sensitivity of the edge detection algorithm. Default value is 0.1.
        lower (int): Optional lower threshold value for the Canny algorithm. If not specified, a value based on sigma
                     and the median pixel value of the image is computed automatically.
        upper (int): Optional upper threshold value for the Canny algorithm. If not specified, a value based on sigma
                     and the median pixel value of the image is computed automatically.

    Returns:
        ndarray: Binary edge map from Canny algorithm.
    """

    mid = np.median(img)

    lower = int(max(0, (1.0 - sigma) * mid)) if lower is None else lower
    upper = int(min(255, (1.0 + sigma) * mid)) if upper is None else upper

    return cv2.Canny(img, threshold1=lower, threshold2=upper)


def equalize_img(img, grid_size=8):
    """
    Applies contrast-limited adaptive histogram equalization to enhance image contrast.

    Args:
        img (ndarray): Input image.
        grid_size (int): Size of the tiles used for adaptive equalization. Default value is 8.

    Returns:
        ndarray: Image with enhanced contrast.
    """

    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(grid_size, grid_size)).apply(img)
