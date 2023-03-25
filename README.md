# README

## proc_img.py

`proc_img.py` is a Python module for processing images using OpenCV. It provides the following functions:

### Functions

- `window_img`: Applies windowing and rescaling to a DICOM image.
- `grayscale_img`: Converts an image to grayscale.
- `otsu_img`: Applies Otsu's thresholding to an image.
- `edge_img`: Applies edge detection to an image using the Canny algorithm.
- `equalize_img`: Applies contrast-limited adaptive histogram equalization to enhance image contrast.

### Classes
- `ProcImg`: A collection of images with methods for batch processing.

## test.py

`test.py` is a Python script that demonstrates the usage of `proc_img.py`. It imports `os`, `glob`, `imageio`, `pandas`, `matplotlib`, and `proc_img`, and defines a `main` function that performs batch processing of DICOM images and creates a GIF animation.

The script reads gaze data from CSV files and extracts window width and level values. Then, it creates an instance of the ProcImg class for each image and applies windowing using the extracted values. Finally, it saves the processed images as PNG files and creates a GIF animation using the `imageio` library.

## Requirements

- Python 3
- OpenCV
- imageio
- pandas
- matplotlib