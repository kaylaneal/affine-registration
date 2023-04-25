# Rigid Registration of 2D WSI Images
- Using 'Perfect Pairs' dataset, containing 8 sets of 100 pairs of source and moving images. Perfect Pairs denotes the same image being used for both inputs, the difference being the rotation and translation of the moving image. Labels for each set is stored in `.json` format.

`dataset.py`
- defines the `PerfectPairsDataset` class, which:
    1. parses json information into image pairs and corresponding labels, and 
    2. contains normalization, segmentation, and other preprocessing functionalities

`affine.py`
- runs registration and visualizaton

`network.py`
- creates the `Affine_Network` Model
    - composed of: regression network, feature extractor, 'forward' block

Initially inspired by DeepHistReg, as defined in Learning-Based Affine Registration of Histological Images.
- [GitHub](https://github.com/MWod/DeepHistReg)
- [Affine Registration Paper](https://link.springer.com/chapter/10.1007/978-3-030-50120-4_2)
- Subsequent Papers: [i.](https://link.springer.com/chapter/10.1007/978-3-030-59861-7_49) and [ii.](https://www.sciencedirect.com/science/article/pii/S0169260720316321)


<br>

# Learning-Based Affine Registration of Histological Images; M. Wodzinski, H. Muller 2020
Proposes a deep network to calculate the initial affine transform between histological images with different dyes. Achieved via a patch-based feature extraction with a variable batch size followed by a 3D convolution combining patch features and 2D convolutions to enlarge the receptive field.

<br>

1. **Preprocessing**
    - Smoothing and Resampling to lower resolution
    - Segment tissue from background
    - Convert image to grayscale
    - Find initial rotation angle

2. **Affine Registration Network**
    - Images passed into network independently
        - unfolded to a grid of *non-overlapping* patches
        - patches combined to a single tensor where \# patches = batch size
    - Feature Extraction by modified ResNet architecture
        - weights shared between source and target
        - features concatenated and passed through additional 2D convolutions to combine to a single representation
    - Global Correspondence is extracted by a 3D convolution
        - followed by 2D convolutions to retrieve global information from unfolded patches
    - Features passed to Adaptive Average Pooling and Fully Connected layers to output Transformation Matrix

<br>

**Algorithm**
- *Input:* $M_p$, $F_p$ (image paths)
- *Output:* $T$ (affine transformation matrix)
    1. $M$, $F$ = load images from $M_p$ and $F_p$
    2. $M$, $F$ = smooth and resample
    3. $M$, $F$ = segment 
    4. $M$, $F$ = convert to grayscale and invert intensities
    5. $T_{rot}$ = find inition rotation angle
        - iteratively by maximizing NCC similarity metric
    6. $M_{rot}$ = warp $M$ with $T_{rot}$
    7. $T_{affine}$ = pass $M_{rot}$ and $F$ through the Affine Network
    8. $T$ = $T_{rot} \cdot T_{affine}$
    9. Return $T$

<br>

*Note on training scheme*
- Image pairs are given one by one
- Loss is backwarded after each pair
- Optimizer is updated only after a gradient of a given number of images are backpropagated
- Patch approach requires replacing batch normalization layers with group normalization
- Adam Optimizer
- Global NCC as cost function