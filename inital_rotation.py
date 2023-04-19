# IMPORTS
import numpy as np
import tensorflow as tf
import skimage.transform as skit

## LOCAL IMPORTS
import utils


def find_translation(source, target):
    source_com_x, source_com_y = utils.center_of_mass(source)
    target_com_x, target_com_y = utils.center_of_mass(target)

    deltaX = source_com_x - target_com_x
    deltaY = source_com_y - target_com_y

    return deltaX, deltaY
    
def apply_translation(source, dx, dy):
    tmat = skit.SimilarityTransform(translation = (dx, dy))
    source = np.array(source).reshape(source.shape[1], source.shape[2])
    tran_source = skit.warp(source, tmat)

    return tran_source
    

def find_theta(source, target):
    angle_step = 45
    min_out_size = 128
    new_shape = utils.calculate_new_shape_min((source.shape[1], source.shape[2]), min_out_size)
    ncc = utils.ncc_loss_global
    gaussian = utils.gaussian_filter(7, 1, 1)

    smooth_source = gaussian(source)
    smooth_target = gaussian(target)

    resample_source = tf.image.resize(smooth_source, new_shape)
    resample_target = tf.image.resize(smooth_target, new_shape)

    init_ncc = ncc(resample_source, resample_target)
    identity = np.array([
        [1, 0, 0.0],
        [0, 1, 0.0],
    ])
    if init_ncc < -0.85:
        return identity
    
    deltaX, deltaY = find_translation(source, target)
    centroid = np.array([
        [1, 0, deltaX],
        [0, 1, deltaY]
    ])
    translated_source = apply_translation(resample_source, deltaX, deltaY)
    translated_source = np.expand_dims(translated_source, (0, -1))
    #print(translated_source.shape)
    centroid_ncc = ncc(translated_source, resample_target)

    if centroid_ncc < 0.85:
        return centroid
    
    best_ncc = centroid_ncc
    found = False
    resample_source = np.array(resample_source).reshape(resample_source.shape[1], resample_source.shape[2])
    for t in range(0, 360, angle_step):
        T = skit.SimilarityTransform(rotation = t, translation = (deltaX, deltaY))
        transformed_source = skit.warp(resample_source, T)
        transformed_source = np.expand_dims(transformed_source, (0, -1))
        current_ncc = ncc(transformed_source, resample_target)

        if current_ncc < best_ncc:
            found = True
            best_ncc = current_ncc
            best_transform = T
    
    if found:
        return best_transform
    elif centroid_ncc < init_ncc:
        return centroid
    else:
        return identity