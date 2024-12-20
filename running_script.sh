# 3D-U-Net
python3 run.py -method=unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
