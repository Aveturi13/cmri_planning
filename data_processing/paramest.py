#Import essential packages
import numpy as np
import cv2
from matplotlib import pyplot as plt

#Mask Generator function
def mask_generator(shape, intercept, slope, width=5):
  """
  Args:
  - shape = tuple containing (height, width) of an image
  - intercept = the y-intercept of the line.
  - slope = the gradient/slope of the line.
  - width = thickness of the line to be drawn.

  Output:
  - A binary mask of shape (height, width) where line coordinates=1 and background=0.
  """

  #Extract height, width, intercept and slope
  h, w = shape
  b, m = intercept, slope

  #Points on line
  x1 = 0
  x2 = 160
  y1 = int(m*x1 + b)
  y2 = int(m*x2 + b)

  mask = np.zeros((h, w))
  mask = cv2.line(mask, (x1, y1), (x2, y2), 1, width)

  return mask

#Loading the dataset
def load_data(image_filepaths, image_params, resize_dim=(160, 160)):

  '''
  image_filepaths = list of file paths for images in dataset
  image_params = matrix of shape (#files, #params) where #params=4 -> (x_values, y_values, dx_values, dy_values)
  '''
  
  #Load the m and b values
  x_values = image_params[:, 0]
  y_values = image_params[:, 1]
  dx_values = image_params[:, 2]
  dy_values = image_params[:, 3]

  #Should be of shape (#files, 160, 160, 1)
  images = []

  #Should be of shape (#files, 160, 160, 1)
  masks = []

  cache = []

  for i in range(len(image_filepaths)):
    #Read image into array
    img = plt.imread(image_filepaths[i])
    (H, W) = img.shape

    #Resize image into 160,160
    img = cv2.resize(img, resize_dim)
    (h, w) = img.shape

    #Append to images list
    images.append(img)

    #Transform line parameters
    x = x_values[i] * w/W
    y = y_values[i] * h/H
    dx = dx_values[i] * w/W
    dy = dy_values[i] * h/H

    #Compute m and b
    m = dy/dx
    b = y - m*x

    #Load masks
    msk = mask_generator(img.shape, b, m, 10)
    masks.append(msk)

    cache.append([m, b])

  #Make shape of images (160, 160, 1)
  images = np.expand_dims(np.array(images), axis=-1)
  masks = np.expand_dims(np.array(masks), axis=-1)

  cache = np.array(cache)

  return images, masks, cache