from re import I
import os
from utils import utils
import tensorflow as tf
from tensorflow import reduce_sum
import tensorflow_probability as tfp
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, UpSampling3D, Conv3DTranspose, BatchNormalization, Add, Activation, Dropout, Lambda, Multiply, Flatten, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class ExpendAsLayer(Layer):
    def __init__(self, rep, **kwargs):
        super(ExpendAsLayer, self).__init__(**kwargs)
        self.rep = rep

    def call(self, inputs):
        return K.repeat_elements(inputs, self.rep, axis=4)

    def get_config(self):
        config = super().get_config()
        config.update({'rep': self.rep})
        return config

class Channel_attention(tf.keras.layers.Layer):
    """ 
    Channel attention module 
    
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(
        self,
        gamma_initializer=tf.zeros_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(Channel_attention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(Channel_attention, self).get_config()
        config.update(
            {
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            }
        )
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4])
        )(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape(
            (input_shape[1], input_shape[2], input_shape[3], input_shape[4])
        )(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs

class Position_attention(tf.keras.layers.Layer):
    """ 
    Position attention module 
        
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(
        self,
        ratio=8,
        gamma_initializer=tf.zeros_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(Position_attention, self).__init__(**kwargs)
        self.ratio = ratio
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(Position_attention, self).get_config()
        config.update(
            {
                'ratio': self.ratio,
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            }
        )
        return config

    def build(self, input_shape):
        super(Position_attention, self).build(input_shape)
        self.query_conv = tf.keras.layers.Conv3D(
            filters=input_shape[-1] // self.ratio,
            kernel_size=(1, 1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.key_conv = tf.keras.layers.Conv3D(
            filters=input_shape[-1] // self.ratio,
            kernel_size=(1, 1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.value_conv = tf.keras.layers.Conv3D(
            filters=input_shape[-1],
            kernel_size=(1, 1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4] // self.ratio)
        )(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4] // self.ratio)
        )(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_key, proj_query)
        attention = tf.keras.activations.softmax(energy)

        proj_value = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4])
        )(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(attention, proj_value)
        outputs = tf.keras.layers.Reshape(
            (input_shape[1], input_shape[2], input_shape[3], input_shape[4])
        )(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs

class Focus_UNet:
  """
  Class for Deep Learning Hyperspectral Segmentation with an architecture similar to Focus U-Net: A novel dual attention-gated CNN for polyp segmentation during colonoscopy (2021).
  
  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils
    self.data_path = os.path.join(os.getcwd(), '')
  
  def encoding_layers_building_blocks(self, units, in_layer, pooling_layer=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      if pooling_layer == True:
          out_layer = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer)
          return out_layer, convolution_layer
      else:
          return convolution_layer
    
  def decoding_layers_building_blocks(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      if concat_layer.shape[3] == 1:
          upsampling_layer = Conv3DTranspose(units, (3, 3, 3), strides=(2, 2, 1), padding='same', kernel_initializer='he_normal')(in_layer)
      else:
          upsampling_layer = Conv3DTranspose(units, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(in_layer)
      upsampling_layer = concatenate([upsampling_layer, concat_layer], axis=4)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      out_layer = Activation('relu')(convolution_layer)
      return out_layer
    
  def focus_layers_building_blocks(self, units, x, g):
      '''
      This method returns the focus layer in the 2021 paper, however, the paper has made a small mistake that upsamples twice the input, we corrected it here.
      '''
      shape_x = K.int_shape(x)
      shape_g = K.int_shape(g)
      
      # Getting the gating signal to the same number of filters as the inter_shape
      phi_g = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(g)

      # Getting the x signal to the same shape as the gating signal
      theta_x = Conv3D(units, (1, 1, 1), strides=(shape_x[1]//shape_g[1], shape_x[2]//shape_g[2], shape_x[3]//shape_g[3]), padding='same', kernel_initializer='he_normal')(x)

      # Element-wise addition of the gating and x signals
      add_xg = Add()([phi_g, theta_x])
      add_xg = Activation('relu')(add_xg)

      # channel attention
      channel_attention_ = Channel_attention()(add_xg)

      # spatial attention
      spatial_attention_ = Position_attention()(add_xg)
      
      # combine channel and spatial weights
      weights = Multiply()([channel_attention_, spatial_attention_])
      
      shape_weight = K.int_shape(weights)
      # Upsampling psi back to the original dimensions of x signal
      upsample_sigmoid_xg = Conv3DTranspose(units, (shape_x[1] // shape_weight[1], shape_x[2] // shape_weight[2], shape_x[3] // shape_weight[3]), strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2], shape_x[3] // shape_g[3]), padding='same', kernel_initializer='he_normal')(weights)

      # Element-wise multiplication of attention coefficients back onto original x signal
      focus_coefficients = Multiply()([upsample_sigmoid_xg, x])

      # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
      output = Conv3D(shape_x[4], (1, 1, 1), strides=(1, 1, 1), padding='same', kernel_initializer='he_normal')(focus_coefficients)
      out_layer = BatchNormalization()(output)
      return out_layer
  
  def dice_loss(self, y_true, y_pred, smooth=1e-6):
    # Calculate intersection and the sum for the numerator and denominator of the Dice score
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_true_pred = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2])

    # Calculate the Dice score for each class
    dice_scores = (2. * intersection + smooth) / (sum_true_pred + smooth)
    dice_loss_multiclass = 1 - K.mean(dice_scores)
    return dice_loss_multiclass
      
  def build_3d_focus_unet(self):
    '''
    This function is a tensorflow realization of the 3D Focus U-Net model (2021). 
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    # down sampling blocks
    down_sampling_output_layer_1, down_sampling_convolution_layer_1 = self.encoding_layers_building_blocks(64, input_layer)
    down_sampling_output_layer_2, down_sampling_convolution_layer_2 = self.encoding_layers_building_blocks(128, down_sampling_output_layer_1)
    down_sampling_output_layer_3, down_sampling_convolution_layer_3 = self.encoding_layers_building_blocks(256, down_sampling_output_layer_2)
    down_sampling_output_layer_4, down_sampling_convolution_layer_4 = self.encoding_layers_building_blocks(512, down_sampling_output_layer_3)
    # encoding blocks
    encoding_space_output_layer = self.encoding_layers_building_blocks(1024, down_sampling_output_layer_4, pooling_layer=False)
    # up sampling blocks
    up_sampling_focus_layer_1 = self.focus_layers_building_blocks(512, down_sampling_convolution_layer_4, encoding_space_output_layer)
    up_sampling_output_layer_1 = self.decoding_layers_building_blocks(512, encoding_space_output_layer, up_sampling_focus_layer_1)
    up_sampling_focus_layer_2 = self.focus_layers_building_blocks(256, down_sampling_convolution_layer_3, encoding_space_output_layer)
    up_sampling_output_layer_2 = self.decoding_layers_building_blocks(256, up_sampling_output_layer_1, up_sampling_focus_layer_2)
    up_sampling_focus_layer_3 = self.focus_layers_building_blocks(128, down_sampling_convolution_layer_2, encoding_space_output_layer)
    up_sampling_output_layer_3 = self.decoding_layers_building_blocks(128, up_sampling_output_layer_2, up_sampling_focus_layer_3)
    up_sampling_focus_layer_4 = self.focus_layers_building_blocks(64, down_sampling_convolution_layer_1, encoding_space_output_layer)
    up_sampling_output_layer_4 = self.decoding_layers_building_blocks(64, up_sampling_output_layer_3, up_sampling_focus_layer_4)
    # classification block
    output_layer = Conv3D(1, (1, 1, 1))(up_sampling_output_layer_4)
    output_layer = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep))(output_layer)
    output_layer = Activation('softmax', name='output_layer')(output_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': self.dice_loss}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    return model

  def train(self):
    '''
    This method internally train the U-Net defined above, with a given set of hyperparameters.
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
        custom_objects = {'dice_loss': self.dice_loss, 'ExpendAsLayer': ExpendAsLayer, 'Position_attention': Position_attention, 'Channel_attention': Channel_attention}
        # Load the full model, including optimizer state
        focus_unet = load_model('saved_models/focus_unet_best_model.h5', custom_objects=custom_objects)
        focus_unet.summary()
        if self.utils.override_trained_optimizer:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
            focus_unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': self.dice_loss}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    else:
        focus_unet = self.build_3d_focus_unet()
        focus_unet.summary()
    print("Training Begins...")
    focus_unet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("saved_models/focus_unet_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    focus_unet = self.build_3d_focus_unet()
    focus_unet.load_weights('saved_models/focus_unet_best_model.h5')
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
      prediction_result = focus_unet.predict(X_test)
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
        prediction_result[i:i+self.utils.batch_size] = focus_unet.predict(self.utils.X_test[i:i+self.utils.batch_size])

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
