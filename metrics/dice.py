import tensorflow.keras
from keras import backend as K

def dice_score(y_true, y_pred, axis=None, smooth=1):

  #Convert predicted mask to a one-hot encoding
  y_true = K.round(y_true)
  y_pred = K.round(y_pred)

  #Calculate DICE score
  intersection = K.sum(y_true * y_pred, axis=axis)
  area_true = K.sum(y_true, axis=axis)
  area_pred = K.sum(y_pred, axis=axis)

  return (2 * intersection + smooth)/(area_true + area_pred + smooth)