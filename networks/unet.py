#Import packages
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Cropping2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

#Downsampling Block
def downsampling_block(input_tensor, filters, padding='same', batchnorm=False, dropout=0.0, reg=l2(l=0.01)):

  """
  Performs a downsampling in the UNet which consists of 2 convolutions and 1 Maxpool.

  Args: 
  - input_tensor = tensor input of shape (h, w, c)
  - filters = number of filters
  - padding = 'same' or 'valid'
  - batchnorm = whether to perform batch normalization or not. 
  - dropout = whether to perform dropout regularization.

  Output: Downsampled tensor of shape (h', w', c')
  """

  #Downsampling
  x = Conv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=reg)(input_tensor)
  x = BatchNormalization()(x) if batchnorm else x
  x = Activation('relu')(x)
  x = Dropout(dropout)(x) if dropout > 0.0 else x

  x = Conv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=reg)(input_tensor)
  x = BatchNormalization()(x) if batchnorm else x
  x = Activation('relu')(x)
  x = Dropout(dropout)(x) if dropout > 0.0 else x

  return MaxPooling2D(pool_size=(2,2))(x), x

def upsampling_block(input_tensor,
                     skip_tensor,
                     filters,
                     padding='same',
                     batchnorm=False,
                     dropout=0.0,
                     reg=l2(l=0.01)):
  """
  Performs an upsampling in the UNet which consists of 1 transpose convolution and 2 convolutions.

  Args: 
  - input_tensor = tensor input of shape (h, w, c)
  - filters = number of filters
  - skip_tensor = the tensor that is concatenated with the current tensor
  - padding = 'same' or 'valid'
  - batchnorm = whether to perform batch normalization or not. 
  - dropout = whether to perform dropout regularization.

  Output: Upsampled tensor of shape (h', w', c')
  """

  #Run an upconvolution
  x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='same', kernel_regularizer=reg)(input_tensor)

  #In order to perform a concatenation, x and skip_tensor must be same dimensions
  _, x_height, x_width, _ = K.int_shape(x)
  _, s_height, s_width, _ = K.int_shape(skip_tensor)
  h_crop = s_height - x_height
  w_crop = s_width - x_width

  assert h_crop >= 0
  assert w_crop >= 0

  if h_crop == 0 and w_crop ==0:
    y = skip_tensor
  else:
    cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
    y = Cropping2D(cropping=cropping)(skip_tensor)

  #Concatenate
  x = Concatenate()([x, y])

  #Run a 2*(conv with kernel (3,3) followed by relu activation)
  x = Conv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=reg)(x)
  x = BatchNormalization()(x) if batchnorm else x
  x = Activation('relu')(x)
  x = Dropout(dropout)(x) if dropout > 0 else x

  
  x = Conv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=reg)(x)
  x = BatchNormalization()(x) if batchnorm else x
  x = Activation('relu')(x)
  x = Dropout(dropout)(x) if dropout > 0 else x

  return x

def unet(filters=32, 
         size=(160, 160, 1), 
         depth=4, 
         padding='same', 
         batchnorm=False, 
         dropout=0.0,
         reg=l2(0.01)):
 
  #Initialize inputs
  x = Input(size)
  inputs = x
  
  #append the operations before the maxpool to this list
  skips = []
  
  #DOWNSAMPLING LAYERS
  for i in range(depth):
    x, x0 = downsampling_block(x, filters, padding, batchnorm, dropout, reg=reg) 
    skips.append(x0)
    filters *= 2
  
  #BOTTLENECK LAYER
  x = Conv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=reg)(x)
  x = BatchNormalization()(x) if batchnorm else x
  x = Activation('relu')(x)
  x = Dropout(dropout)(x) if dropout > 0 else x

  x = Conv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=reg)(x)
  x = BatchNormalization()(x) if batchnorm else x
  x = Activation('relu')(x)
  x = Dropout(dropout)(x) if dropout > 0 else x
  
  #UPSAMPLING LAYERS
  for i in reversed(range(depth)):
    filters //= 2
    x = upsampling_block(x, skips[i], filters, padding, batchnorm, dropout, reg=reg)
    
  outputs = Conv2D(1, 1, activation='sigmoid')(x)
  
  #model creation 
  model = Model(inputs=[inputs], outputs=[outputs])
  
  return model