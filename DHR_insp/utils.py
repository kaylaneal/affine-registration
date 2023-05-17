# IMPORTS
import numpy as np
import scipy.ndimage as nd

## PREPROCESSING STEPS
def normalize(image: np.array):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def resample_image(image, resample_ratio):
    y_size, x_size = image.shape[1], image.shape[0]
    new_y, new_x = int(y_size / resample_ratio), int(x_size / resample_ratio)

    grid_x, grid_y = np.meshgrid(np.arange(new_x), np.arange(new_y))
    grid_x = grid_x * (x_size / new_x)
    grid_y = grid_y * (y_size / new_y)
    resampled = nd.map_coordinates(image, [grid_y, grid_x], cval = 0, order = 3)
    return resampled

def pad_images(source, target):
    y_source, x_source = source.shape
    y_target, x_target = target.shape

    new_y = max(y_source, y_target)
    new_x = max(x_source, x_target)

    padded_source = pad_single(source, (new_y, new_x))
    padded_target = pad_single(target, (new_y, new_x))
    return padded_source, padded_target

def pad_single(image, new_shape):
    y, x = image.shape
    y_pad = ((int(np.floor((new_shape[0] - y) / 2))), int(np.ceil((new_shape[0] - y) / 2)))
    x_pad = ((int(np.floor((new_shape[1] - x) / 2))), int(np.ceil((new_shape[1] - x)) / 2))
    new_image = np.pad(image, (y_pad, x_pad), constant_values = 0)

    return new_image



## Segmentation Utilities
def calculate_new_shape_min(current_shape, min_size):
    if current_shape[0] > current_shape[1]:
        divider = current_shape[1] / min_size
    else:
        divider = current_shape[0] / min_size
    
    new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))
    return new_shape