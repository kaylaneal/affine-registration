# IMPORTS
import os
import numpy as np
import scipy.ndimage as nd
from tensorflow.keras import layers
import tensorflow as tf

## LOCAL IMPORTS


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

## COST FUNCTIONS

def ncc_loss_global(source, target):
    s_view = np.reshape(source, (1, 1, source.shape[1], source.shape[2]))
    t_view = np.reshape(target, (1, 1, target.shape[1], target.shape[2]))
    return ncc_global(s_view, t_view)

def ncc_global(source, target):
    size = source.shape[2] * source.shape[3]

    source_mean = np.mean(source, axis = (1, 2, 3))
    source_mean = np.reshape(source_mean, (source.shape[0], 1, 1, 1))
    target_mean = np.mean(target, axis = (1, 2, 3))
    target_mean = np.reshape(target_mean, (source.shape[0], 1, 1, 1))

    source_std = np.std(source, axis = (1, 2, 3))
    source_std = np.reshape(source_std, (source.shape[0], 1, 1, 1))
    target_std = np.std(target, axis = (1, 2, 3))
    target_std = np.reshape(target_std, (source.shape[0], 1, 1, 1))

    ncc = (1 / size) * np.sum((source - source_mean) * (target - target_mean) / (source_std * target_std), axis = (1, 2, 3))
    return ncc

def dice_loss(prediction, target):
    smooth = 1

    prediction = np.reshape(prediction, -1)
    target = np.reshape(target, -1)

    intersect = np.sum(prediction * target)
    return 1 - ((2 * intersect + smooth) / (np.sum(prediction) + np.sum(target) + smooth))

## Segmentation Utilities
def calculate_new_shape_min(current_shape, min_size):
    if current_shape[0] > current_shape[1]:
        divider = current_shape[1] / min_size
    else:
        divider = current_shape[0] / min_size
    
    new_shape = (int(current_shape[0] / divider), int(current_shape[1] / divider))
    return new_shape

def LoG(kernel_size = 3, sigma = 2, channels = 3):
    # Create Grid
    x = np.arange(kernel_size)
    x = np.tile(x, kernel_size).reshape(kernel_size, kernel_size)
    y = x.transpose()
    grid = np.stack([x, y], axis = -1)

    # Statistics Calculations
    mean = kernel_size / 2
    var = sigma**2

    LoG = ( (np.sum((grid - mean)**2., axis = -1) * var) /
           (2 * np.pi * (sigma**6)) *
            np.exp(-np.sum((grid - mean)**2., axis = -1) / (2 * var)) )
    LoG_filter = layers.Conv2D(channels, kernel_size, groups = channels, use_bias = False, padding = 'same')
    LoG_filter.weights.append(LoG)
    return LoG_filter

def segmentation(source, target):
    output_min_size = 128
    new_shape = calculate_new_shape_min((source.shape[1], source.shape[2]), output_min_size)
    resample_source = tf.image.resize(source, new_shape, preserve_aspect_ratio = True)
    resample_target = tf.image.resize(target, new_shape, preserve_aspect_ratio = True)

    LoG_filter = LoG(7, 2, 1)
    source_mask = LoG_filter(resample_source)
    target_mask = LoG_filter(resample_target)

    #print(f'S Mask Shape: {source_mask.shape}\nT Mask Shape: {target_mask.shape}')

    source_mask = tf.image.resize(source_mask, (source.shape[1], source.shape[2]), preserve_aspect_ratio = True) > 0.5
    target_mask = tf.image.resize(target_mask, (target.shape[1], target.shape[2]), preserve_aspect_ratio = True) > 0.5

    source_mask = tf.cast(source_mask, tf.float32)
    target_mask = tf.cast(target_mask, tf.float32)

    return source_mask, target_mask

## inital theta
def center_of_mass(image):
    if len(image.shape) == 2:
        x, y = image.shape
    elif len(image.shape) == 3:
        x, y = image.shape[0], image.shape[2]
    elif len(image.shape) == 4:
        x, y = image.shape[1], image.shape[3]
    else:
        raise NotImplementedError
    gy, gx = np.meshgrid(np.arange(y), np.arange(x))

    m00 = np.sum(image)
    m10 = np.sum((gx * image))
    m01 = np.sum((gy * image))

    com_x = m10 / m00
    com_y = m01 / m00
    return com_x, com_y

def gaussian_filter(kernel_size = 3, sigma = 2, channels = 3):
    # Create Grid:
    x = np.arange(kernel_size)
    x = np.tile(x, kernel_size).reshape(kernel_size, kernel_size)
    y = x.transpose()
    grid = np.stack([x, y], axis = -1)

    # Statistics Calculations:
    mean = kernel_size / 2
    var = sigma**2

    # Compute Gaussian Distribution:
    gk = (1.0 / (2.0 * np.pi * var)) * np.exp(-np.sum(grid - mean)**2 / (2 * var))
    
    # Create Convolution Matrix where Gaussian Distribution is the Weight
    gfilter = layers.Conv2D(channels, kernel_size = kernel_size,
                            groups = channels, use_bias = False)
    gfilter.weights.append(gk)
    return gfilter

def affine2theta(affine, shape):
    h, w = shape
    theta = np.zeros((2, 3))
    temp = affine

    theta[0, 0] = temp[0, 0]
    theta[0, 1] = temp[0, 1] * (h/w)
    theta[0, 2] = temp[0, 2] * (2/w) + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = temp[1, 0] * (w/h)
    theta[1, 1] = temp[1, 1]
    theta[1, 2] = temp[1, 2] * (2/h) + theta[1, 0] + theta[1, 1] - 1
    
    return theta

def theta2affine(theta, shape):
    h, w = shape
    affine = np.zeros(2, 3)
    temp = theta

    affine[1, 2] = (temp[1, 2] - temp[1, 0], - temp[1, 1] + 1) * (h/2)
    affine[1, 1] = temp[1, 1]
    affine[1, 0] = temp[1, 0] * (h/w)
    affine[0, 2] = (temp[0, 2] - temp[0, 1] - temp[0, 0] + 1) * (w/2)
    affine[0, 1] = temp[0, 1] * (w/h)
    affine[0, 0] = temp[0, 0]

    return affine

def compose_transform(t1, t2):
    tr1 = np.zeros((3, 3))
    tr2 = np.zeros((3, 3))
    tr1[0:2, :] = t1
    tr2[0:2, :] = t2
    tr1[2, 2] = 1
    tr2[2, 2] = 1
    result = np.matmul(tr1, tr2)
    return result[0:2, :]

def generate_rotation_matrix(angle, x0, y0):
    angle = angle * np.pi / 180
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    cm1 = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ])

    transformation = cm1 @ R @ cm2
    return transformation[0:2, :]
