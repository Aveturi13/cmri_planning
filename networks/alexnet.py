#Import modules and packages
import keras
from keras.models import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Cropping2D, Input, Concatenate, BatchNormalization, Dropout, Dense, Flatten, AveragePooling2D, Add
from keras.initializers import glorot_uniform
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf

def alexnet(input_size=(160, 160, 1), batchnorm=False, dropout=0.0, optimize='both'):

  #Initialize input
  x = Input(input_size)
  inputs = x

  #Convolutional layers
  x = Conv2D(filters=96, kernel_size=(11,11), strides=4)(x)
  x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Conv2D(filters=256, kernel_size=(5,5), padding='same')(x)
  x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Conv2D(filters=384, kernel_size=(3,3), padding='same')(x)
  x = Conv2D(filters=384, kernel_size=(3,3))(x)
  x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Flatten()(x)

  x = Dense(units=384, activation='relu')(x)
  x = Dropout(rate=dropout)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Dense(units=96, activation='relu')(x)
  x = Dropout(rate=dropout)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Dense(units=48, activation='relu')(x)
  x = Dropout(rate=dropout)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Dense(units=12, activation='relu')(x)
  x = Dropout(rate=dropout)(x)
  x = BatchNormalization()(x) if batchnorm else x

  x = Dense(units=6, activation='relu')(x)
  x = Dropout(rate=dropout)(x)
  x = BatchNormalization()(x) if batchnorm else x

  if optimize == 'r' or optimize == 'theta':
    outputs = Dense(units=1, activation='sigmoid')(x)
  elif optimize == 'both':
    outputs = Dense(units=2, activation='sigmoid')(x)

  model = Model(inputs=[inputs], outputs=[outputs])
  
  return model