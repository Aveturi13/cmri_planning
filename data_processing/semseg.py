#Import essential packages
import os, glob
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import time

from .maskgen import mask_generator

#Get current time
now = time.localtime()
current_date = time.strftime("%d-%m-%y", now)

class Dataset():

  """ 

  Base Dataset class 

  Attributes
  ----------

  img_dir : str
        directory path for the dataset
  params : str
        file path to csv file of 2Ch line values
  image_filepaths : list
        list of all image files in directory img_dir
  len : int
        number of examples in directory
  slopes : np.array
        1D array of slopes for 2Ch line
  intercepts : np.array
        1D array of intercepts for 2Ch line
  params : np.array
        2D array of slopes and intercepts together, shape (len, 2)


  Methods
  -------
  split() : Splits data into train and test sets
  data_generator() : Creates a generator object that can be used for model training
  """

  def __init__(self, img_dir, param_file):
    self.img_dir = img_dir
    self.param_file = param_file
    
    #Put all the image files into a list
    glob_search = os.path.join(self.img_dir, '00*.png')
    self.image_filepaths = sorted(glob.glob(glob_search))
    self.len = len(self.image_filepaths)

    #Extract parameter values from params file
    df = pd.read_csv(self.param_file, header=None)
    line_data = df.iloc[:, 1:]
    self.params = line_data.values

    #Check number of images = number of parameters in csv
    assert self.params.shape[0] == self.len

    #x_values = line_array[:, 0]
    #y_values = line_array[:, 1]
    #dx_values = line_array[:, 2]
    #dy_values = line_array[:, 3]

    #Calculate slope and intercept of lines
    #self.slopes = dy_values/dx_values
    #self.intercepts = y_values - self.slopes * x_values

    #Combine into one params array
    #self.params = np.column_stack((self.slopes, self.intercepts))

  def generate_preview(self, images, params, resize_dim):

    #Create a directory to save the sample images
    if not os.path.exists('images'):
      os.mkdir('images')

    #If previewing a single image only
    if not type(images) == list:
      image_sample = [images]
      param_sample = np.expand_dims(params, 0)

      image, mask = self.preprocess_data(image_sample, param_sample, resize_dim)
      image = image.squeeze()
      mask = mask.squeeze()

      #Save image
      if not os.path.exists('images/{}'.format(current_date)):
        os.mkdir('images/{}'.format(current_date))

      plt.imsave('images/{}/image.jpg'.format(current_date), image, cmap='gray')
      plt.imsave('images/{}/mask.jpg'.format(current_date), mask, cmap='gray')

    # if type(images) == list:
    #   image, mask = self.preprocess_data(images, params, resize_dim)
    #   image = image.squeeze()
    #   mask = mask.squeeze()

  def split(self, shuffle=True, split=0.25):

    """ Splits data into train and test sets"""

    if shuffle:
      state= np.random.get_state()
      img_shuffled = np.random.permutation(self.image_filepaths)
      np.random.set_state(state)
      params_shuffled = np.random.permutation(self.params)
    else:
      img_shuffled = self.image_files
      params_shuffled = self.params

    split_index = int((1-split)*self.len)

    train_images = img_shuffled[:split_index]
    val_images = img_shuffled[split_index:]

    train_params = params_shuffled[:split_index]
    val_params = params_shuffled[split_index:]

    return (train_images, train_params, val_images, val_params)

  def preprocess_data(self, images, params, resize_dim=(160, 160)):

    """ Similar to data_generator, but only preprocesses a small sample of data"""

    #slopes, intercepts = params[:, 0], params[:, 1]
    #print(slope.shape, intercepts.shape)

    images_preprocessed = []
    masks_preprocessed = []

    for i in range(len(images)):

      img = plt.imread(images[i])
      (H, W) = img.shape

      #Resize image into 160,160
      img = cv2.resize(img, resize_dim, cv2.INTER_CUBIC)
      (h, w) = img.shape

      #Transform line parameters
      x = params[i, 0] * w/W
      y = params[i, 1] * h/H
      dx = params[i, 2] * w/W
      dy = params[i, 3] * h/H

      #Compute new slope and intercept
      m = dy / dx
      b = y - m*x

      mask = mask_generator(img.shape, slope=m, intercept=b)

      #Resize image and mask
      #img = cv2.resize(img, resize_dim, cv2.INTER_CUBIC) if resize_dim else img
      #mask = cv2.resize(mask, resize_dim, cv2.INTER_CUBIC) if resize_dim else mask

      images_preprocessed.append(img)
      masks_preprocessed.append(mask)

    images_preprocessed = np.expand_dims(np.array(images_preprocessed), axis=-1)
    masks_preprocessed = np.expand_dims(np.array(masks_preprocessed), axis=-1)

    return (images_preprocessed, masks_preprocessed)

  def data_generator(self, images, params, resize_dim=(160, 160), batch_size=32):

    """ Creates Data Generator objects """

    while True:
      #need to break params down into slopes and intercepts
      #slopes, intercepts = params[:, 0], params[:, 1]

      #Grab a random batch of samples
      rows_to_select = np.random.choice(len(images), batch_size, replace=False)
      image_batch = [images[i] for i in rows_to_select]
      params_batch = params[rows_to_select, :]

      # slope_batch = np.random.choice(slopes, batch_size)
      # np.random.set_state(state)
      # intercept_batch = np.random.choice(intercepts, batch_size)

      #Processed data will be added here
      image_batch_final = []
      mask_batch_final = []

      for i in range(len(image_batch)):

        #Read image into array
        img = plt.imread(image_batch[i])
        (H, W) = img.shape

        #Resize image into 160,160
        img = cv2.resize(img, resize_dim, cv2.INTER_CUBIC)
        (h, w) = img.shape

        #Transform line parameters
        x = params_batch[i, 0] * w/W
        y = params_batch[i, 1] * h/H
        dx = params_batch[i, 2] * w/W
        dy = params_batch[i, 3] * h/H

        #Compute new slope and intercept
        m = dy / dx
        b = y - m*x

        mask = mask_generator(img.shape, slope=m, intercept=b)

        #Resize image and mask
        img = cv2.resize(img, resize_dim, cv2.INTER_CUBIC) if resize_dim else img
        mask = cv2.resize(mask, resize_dim, cv2.INTER_CUBIC) if resize_dim else mask

        image_batch_final.append(img)
        mask_batch_final.append(mask)

      image_batch_final = np.expand_dims(np.array(image_batch_final), axis=-1)
      mask_batch_final = np.expand_dims(np.array(mask_batch_final), axis=-1)

      yield (image_batch_final, mask_batch_final)