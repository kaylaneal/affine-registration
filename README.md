# Rigid Registration of 2D WSI Images

- Using 'Perfect Pairs' dataset, containing 8 sets of 100 pairs of source and target images. 
    - Perfect Pairs denotes the same image being used for both inputs, the difference being the rotation and translation of the source image. 
    - Labels for each set is stored in `.json` format.

## Objective:

Given a pair of images, one *Source* image and one *Target* image, in which the Source image is a rigidly transformed version of the Target image. Train a *Neural Network* to predict the *transformation parameters* between the two images, or, Find the transformation that maps the Source to the Target image.

$$images = [S, T]$$
$$parameters = [\Delta\Theta, \Delta X, \Delta Y]$$


<br>

Initially inspired by [DeepHistReg](https://github.com/MWod/DeepHistReg), as defined in:
- [Affine Registration Paper](https://link.springer.com/chapter/10.1007/978-3-030-50120-4_2)
- and Subsequent Papers: 

   - [i.](https://link.springer.com/chapter/10.1007/978-3-030-59861-7_49)  
    
   - [ii.](https://www.sciencedirect.com/science/article/pii/S0169260720316321)


<br>

