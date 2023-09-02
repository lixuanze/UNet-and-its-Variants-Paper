# UNet-and-its-Variants-Paper-Codebase

This is a sub-module of the BHSIpy.

## I. Sample Usage of the Deep Learning Solver

### 3D-UNet
```python 
python3 run.py -method=unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=False -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=True
```
