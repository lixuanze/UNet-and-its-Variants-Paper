from re import I
import os
from utils import utils
import tensorflow as tf
from tensorflow import reduce_sum
import tensorflow_probability as tfp
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, UpSampling3D, BatchNormalization, Add, Activation, Dropout, Flatten, Lambda, Multiply, Layer, Conv3DTranspose
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class Residual_UNet_plus_plus:
  """
  Class for Deep Learning Hyperspectral Segmentation with ResUNet++: An Advanced Architecture for Medical Image Segmentation

  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils
    self.data_path = os.path.join(os.getcwd(), '')
    
  def encoding_layers_building_blocks(self, units, in_layer, downsampling=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      shortcut = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer = BatchNormalization()(in_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = Add()([convolution_layer, shortcut]) # Residual Connections
      if downsampling:
          output_layer = Conv3D(units, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(convolution_layer)
          return output_layer, convolution_layer
      else:
          return convolution_layer
    
  def decoding_layers_building_blocks(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      if concat_layer.shape[3] == 1:
          upsampling_layer = UpSampling3D(size=(2, 2, 1))(in_layer)
          upsampling_layer = concatenate([upsampling_layer, concat_layer], axis=4)
      else:
          upsampling_layer = UpSampling3D(size=(2, 2, 2))(in_layer)
          upsampling_layer = concatenate([upsampling_layer, concat_layer], axis=4)
      convolution_layer = BatchNormalization()(upsampling_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      shortcut = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      out_layer = Add()([convolution_layer, shortcut]) # Residual Connections
      return out_layer
    
  def dice_loss(self, y_true, y_pred, smooth=1e-6):
    # Calculate intersection and the sum for the numerator and denominator of the Dice score
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_true_pred = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2])

    # Calculate the Dice score for each class
    dice_scores = (2. * intersection + smooth) / (sum_true_pred + smooth)
    dice_loss_multiclass = 1 - K.mean(dice_scores)
    return dice_loss_multiclass
      
  def build_3d_residual_unet_plus_plus(self):
    '''
    This function is a tensorflow realization of the 3D-Residual-U-Net++ Model (2019). 
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    
    # L1
    down_sampling_pooling_layer_1, encoding_space_layer_1 = self.encoding_layers_building_blocks(64, input_layer)
    down_sampling_pooling_layer_2, encoding_space_layer_2 = self.encoding_layers_building_blocks(128, down_sampling_pooling_layer_1)
    up_output_1 = self.decoding_layers_building_blocks(64, encoding_space_layer_2, encoding_space_layer_1)
    
    # L2
    down_sampling_pooling_layer_3, encoding_space_layer_3 = self.encoding_layers_building_blocks(256, down_sampling_pooling_layer_2)
    up_output_22 = self.decoding_layers_building_blocks(128, encoding_space_layer_3, encoding_space_layer_2)
    concat_layer_21 = concatenate([encoding_space_layer_1, up_output_1], axis=4)
    up_output_2 = self.decoding_layers_building_blocks(64, up_output_22, concat_layer_21)

    # L3
    down_sampling_pooling_layer_4, encoding_space_layer_4 = self.encoding_layers_building_blocks(512, down_sampling_pooling_layer_3)
    up_output_33 = self.decoding_layers_building_blocks(256, encoding_space_layer_4, encoding_space_layer_3)
    concat_layer_32 = concatenate([encoding_space_layer_2, up_output_22], axis=4)
    up_output_32 = self.decoding_layers_building_blocks(128, up_output_33, concat_layer_32)
    concat_layer_31 = concatenate([encoding_space_layer_1, up_output_1, up_output_2], axis=4)
    up_output_3 = self.decoding_layers_building_blocks(64, up_output_32, concat_layer_31)
    
    # L4
    encoding_space_output_layer = self.encoding_layers_building_blocks(1024, down_sampling_pooling_layer_4, downsampling=False)
    up_output_44 = self.decoding_layers_building_blocks(512, encoding_space_output_layer, encoding_space_layer_4)
    concat_layer_43 = concatenate([encoding_space_layer_3, up_output_33], axis=4)
    up_output_43 = self.decoding_layers_building_blocks(256, up_output_44, concat_layer_43)
    concat_layer_42 = concatenate([encoding_space_layer_2, up_output_22, up_output_32], axis=4)
    up_output_42 = self.decoding_layers_building_blocks(128, up_output_43, concat_layer_42)
    concat_layer_41 = concatenate([encoding_space_layer_1, up_output_1, up_output_2, up_output_3], axis=4)
    up_output_4 = self.decoding_layers_building_blocks(64, up_output_42, concat_layer_41)
    
    # classification blocks
    output_layer_1 = Conv3D(1, (1, 1, 1))(up_output_1)
    output_layer_1 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep))(output_layer_1)
    output_layer_1 = Activation('softmax', name='output_layer_1')(output_layer_1)
    
    output_layer_2 = Conv3D(1, (1, 1, 1))(up_output_2)
    output_layer_2 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep))(output_layer_2)
    output_layer_2 = Activation('softmax', name='output_layer_2')(output_layer_2)
    
    output_layer_3 = Conv3D(1, (1, 1, 1))(up_output_3)
    output_layer_3 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep))(output_layer_3)
    output_layer_3 = Activation('softmax', name='output_layer_3')(output_layer_3)
    
    output_layer_4 = Conv3D(1, (1, 1, 1))(up_output_4)
    output_layer_4 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep))(output_layer_4)
    output_layer_4 = Activation('softmax', name='output_layer_4')(output_layer_4)
    
    if self.utils.deep_supervision == True:
        model = Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4])
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_1': self.dice_loss, 'output_layer_2': self.dice_loss, 'output_layer_3': self.dice_loss, 'output_layer_4': self.dice_loss}, metrics={'output_layer_1': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_2': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_3': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    else:
        model = Model(inputs=[input_layer], outputs=[output_layer_4])
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_4': self.dice_loss}, metrics={'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    return model

  def train(self):
    '''
    This method internally train the Residual U-Net++ defined above, with a given set of hyperparameters.
    '''
    if self.utils.pre_load_dataset == True:
      print("Loading Training & Validation Dataset...")
      self.utils.X_train = np.load(os.path.join(self.data_path, 'Data/X_train.npy')).astype(np.float64)
      self.utils.X_validation = np.load(os.path.join(self.data_path, 'Data/X_validation.npy')).astype(np.float64)
      self.utils.y_train = np.load(os.path.join(self.data_path, 'Data/y_train.npy')).astype(np.float64)
      self.utils.y_validation = np.load(os.path.join(self.data_path, 'Data/y_validation.npy')).astype(np.float64)
      print("Training and Validation Dataset Loaded")
    else:
      print("Data Loading...")
      X, y = self.utils.DataLoader(self.utils.dataset)
      print("Data Loading Completed")
      if self.utils.svd_denoising == True:
        print("Raw Data Denoising...")
        denoised_data, variance_matrix = self.utils.svd_denoise(X, n_svd = self.utils.n_svd)
        # Compute cumulative variance explained
        cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
        while True:
          if cumulative_variance[self.utils.n_svd - 1] <= self.utils.svd_denoise_threshold:
              self.utils.n_svd += 1
              denoised_data, variance_matrix = self.utils.svd_denoise(X, n_svd = self.utils.n_svd)
              cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
          else:
              X = denoised_data
              break
        print("Raw Data Denoise Completed")
      else:
        pass
      if self.utils.layer_standardization == True:
        print("Layer-Wise Standardization...")
        X_normalized = self.utils.run_layer_standardization(X)
        print("Layer-Wise Standardization Completed")
      else:
        X_normalized = X
      print("Prepare Data for Training, Validation & Testing...")
      X_processed, y_processed = self.utils.prepare_dataset_for_training(X_normalized, y)
      X_train, X_, y_train, y_ = train_test_split(X_processed, y_processed, train_size = 1 - self.utils.test_ratio, test_size = self.utils.test_ratio, random_state=1234)
      X_validation, X_test, y_validation, y_test = train_test_split(X_, y_, train_size = 1 - self.utils.test_ratio, test_size = self.utils.test_ratio, random_state=4321)
      self.utils.X_train, self.utils.X_validation, self.utils.X_test = X_train, X_validation, X_test
      self.utils.y_train, self.utils.y_validation, self.utils.y_test = y_train, y_validation, y_test
      np.save(os.path.join(self.data_path, 'Data/X_train.npy'), self.utils.X_train)
      np.save(os.path.join(self.data_path, 'Data/X_validation.npy'), self.utils.X_validation)
      np.save(os.path.join(self.data_path, 'Data/X_test.npy'), self.utils.X_test)
      np.save(os.path.join(self.data_path, 'Data/y_train.npy'), self.utils.y_train)
      np.save(os.path.join(self.data_path, 'Data/y_validation.npy'), self.utils.y_validation)
      np.save(os.path.join(self.data_path, 'Data/y_test.npy'), self.utils.y_test)
      print("Data Processing Completed")
    if self.utils.continue_training == True:
        # Custom objects, if any (you might need to define them depending on the custom loss, metrics, etc.)
        custom_objects = {'dice_loss': self.dice_loss}
        # Load the full model, including optimizer state
        attention_unet = load_model('saved_models/residual_unet_plus_plus_best_model.h5', custom_objects=custom_objects)
        attention_unet.summary()
        if self.utils.override_trained_optimizer:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
            if self.utils.deep_supervision == True:
                attention_unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_1': self.dice_loss, 'output_layer_2': self.dice_loss, 'output_layer_3': self.dice_loss, 'output_layer_4': self.dice_loss}, metrics={'output_layer_1': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_2': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_3': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
            else:
                attention_unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_4': self.dice_loss}, metrics={'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    else:
        attention_unet = self.build_3d_residual_unet_plus_plus()
        attention_unet.summary()
    print("Training Begins...")
    attention_unet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("saved_models/residual_unet_plus_plus_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    attention_unet = self.build_3d_residual_unet_plus_plus()
    attention_unet.load_weights('saved_models/residual_unet_plus_plus_best_model.h5')
    if new_data is not None:
      n_features = self.utils.n_features
      if self.utils.svd_denoising == True:
          print("Raw Data Denoising...")
          denoised_data, variance_matrix = self.utils.svd_denoise(new_data, n_svd = self.utils.n_svd)
          # Compute cumulative variance explained
          cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
          while True:
            if cumulative_variance[self.utils.n_svd - 1] <= self.utils.svd_denoise_threshold:
              self.utils.n_svd += 1
              denoised_data, variance_matrix = self.utils.svd_denoise(new_data, n_svd = self.utils.n_svd)
              cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
            else:
              new_data = denoised_data
              break
          print("Raw Data Denoise Completed")
      else:
          pass
      if self.utils.layer_standardization == True:
        print("Layer-Wise Standardization...")
        X_normalized = self.utils.run_layer_standardization(new_data)
        print("Layer-Wise Standardization Completed")
      else:
        X_normalized = new_data
      X_, pca = self.utils.run_PCA(image_cube = X_normalized, num_principal_components = self.utils.num_components_to_keep)
      X_ = cv2.resize(X_, (self.utils.resized_x_y, self.utils.resized_x_y), interpolation = cv2.INTER_LANCZOS4)
      X_test = X_.reshape(-1, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1)
      prediction_result = attention_unet.predict(X_test)[3]
      prediction_encoded = np.zeros((self.utils.resized_x_y, self.utils.resized_x_y))
      for i in range(self.utils.resized_x_y):
        for j in range(self.utils.resized_x_y):
            prediction_encoded[i][j] = np.argmax(prediction_result[0][i][j])

      prediction = cv2.resize(prediction_encoded, (new_data.shape[1], new_data.shape[0]), interpolation = cv2.INTER_NEAREST)
      return prediction
    else:
      if self.utils.pre_load_dataset == True:
        print("Loading Testing Dataset...")
        self.utils.X_test = np.load(os.path.join(self.data_path, 'Data/X_test.npy')).astype(np.float64)
        self.utils.y_test = np.load(os.path.join(self.data_path, 'Data/y_test.npy')).astype(np.float64)
        print("Testing Dataset Loaded")
      else:
        print("Testing Begins...")
      total_test_length = (self.utils.X_test.shape[0]//self.utils.batch_size)*self.utils.batch_size
      prediction_result = np.zeros(shape=(total_test_length, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))
      for i in range(0, total_test_length, self.utils.batch_size):
        print("Testing sample from:", i, "to:", i+self.utils.batch_size)
        prediction_result[i:i+self.utils.batch_size] = attention_unet.predict(self.utils.X_test[i:i+self.utils.batch_size])[3]

      prediction_ = np.zeros(shape=(total_test_length, self.utils.resized_x_y, self.utils.resized_x_y))
      for k in range(total_test_length):
        prediction_encoded = np.zeros((self.utils.resized_x_y, self.utils.resized_x_y))
        for i in range(self.utils.resized_x_y):
          for j in range(self.utils.resized_x_y):
            prediction_encoded[i][j] = np.argmax(prediction_result[k][i][j])
        prediction = cv2.resize(prediction_encoded, (self.utils.X_test[k].shape[1], self.utils.X_test[k].shape[0]), interpolation = cv2.INTER_LANCZOS4)
        prediction_[k] = prediction
      print("Testing Ends...")
      return prediction_, np.argmax(self.utils.y_test[0:total_test_length], axis=-1)