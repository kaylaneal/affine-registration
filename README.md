# Rigid / Affine Registration of 2D Whole Slide Images (WSIs)

## Dataset

**Pefect Pairs Dataset**
- Consists of 8 sets of images.
    - Each image set contains 100 pairs of source and moving images.
    - The source and moving image are the same image, the difference is in the rigid transformation between the images
    - The transformations are *known*
    - Each set has one `info.json` file, containing the image pair and the transformation label

## Objective:

Given a pair of images, one *Source* image and one *Target* image, in which the Source image is a rigidly transformed version of the Target image. Train a *Neural Network* to predict the *transformation parameters* between the two images, or, Find the transformation that maps the Source to the Target image.

$$images = [S, T]$$
$$parameters = [\Delta\Theta, \Delta X, \Delta Y]$$
