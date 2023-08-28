# 3D-U-Net
python3 run.py -method=unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# 3D-U-Net++
python3 run.py -method=unet_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# 3D-U-Net 3+
python3 run.py -method=unet_3plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# 3D-SegNet
python3 run.py -method=segnet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# V-Net
python3 run.py -method=vnet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# 3D-Inception U-Net
python3 run.py -method=inception_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -dropout_rate=0.1 -override_trained_optimizer=False
# 3D-Residual U-Net
python3 run.py -method=residual_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# 3D-Dense U-Net
python3 run.py -method=dense_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
# 3D-R2U-Net
python3 run.py -method=r2unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False
