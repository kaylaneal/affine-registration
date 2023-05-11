# Rigid / Affine Registration of 2D Whole Slide Images (WSIs)

*GOAL*: Create a Convolutional Neural Network (CNN) to accurately predict the rigid transformation between a pair of images.

*BACKGROUND*: Registration tasks aim to map one image into another image's coordiante space. 

## Dataset

**Pefect Pairs Dataset**
- Consists of 8 sets of images.
    - Each image set contains 100 pairs of source and moving images.
    - The source and moving image are the same image, the difference is in the rigid transformation between the images
    - The transformations are *known*
    - Each set has one `.json` file, containing the image pair and the transformation label