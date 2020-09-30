#Import modules
import cv2
import numpy as np

def mask_generator(shape, intercept, slope, width=5):
  """
  Generates a mask given a CMR image.

  Args
  ----
  shape : tuple 
        Tuple of (height, width) of image
  intercept : float
        The y-intercept of 2Ch line
  slope : float
        The gradient/slope of 2Ch line
  width : int
        Thickness of the line on mask

  Output
  ------
  Mask of shape (height, width) where pixels corresponding to line/background store 1/0.
  """

  #Extract height, width, intercept and slope
  h, w = shape
  b, m = intercept, slope

  #Calculate points on line
  x1 = 0
  x2 = h
  y1 = int(m*x1 + b)
  y2 = int(m*x2 + b)

  #Create mask
  mask = np.zeros((h, w))
  mask = cv2.line(mask, (x1, y1), (x2, y2), 1, width)

  return mask