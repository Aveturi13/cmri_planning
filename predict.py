""" Generate predictions on test images """

#Import packages
import os
import numpy as np
import argparse
import time
import cv2
from matplotlib import pyplot as plt

from data_processing import semseg

import tensorflow as tf
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_dir', help='Path to Image directory', type=str)
	parser.add_argument('--params', help='Path to parameters file', type=str)
	parser.add_argument('--model', help='Path to model file', type=str)
	parser.add_argument('--crop_size', default=(160, 160), help='Resizing dimensions')
	parser.add_argument('--save_predictions', action='store_true', help='Saves predictions')
	parser.add_argument('--verbose', action='store_true', help='Verbose')
	args = parser.parse_args()

	#Create dataset and generate (image, target) pairs
	if args.verbose:
		print("## Loading Dataset ##")

	data = semseg.Dataset(args.img_dir, args.params)
	images, true_masks = data.preprocess_data(data.image_filepaths[:10], data.params[:10], args.crop_size)

	#Load model
	if args.verbose:
		print("## Loading Model ##")

	model = load_model(args.model)

	if args.verbose:
		print("## Predicting on test images ##")
	pred_masks = model.predict_on_batch(images)

	pred_folder = args.model.replace('trainedmodels/', "")[:-3]

	if args.save_predictions:
		if args.verbose:
			print("## Saving Predictions ##")

		if not os.path.exists('predictions/'):
			os.mkdir('predictions/')

		if not os.path.exists('predictions/'+pred_folder):
			os.mkdir('predictions/'+pred_folder)

		for i in range(images.shape[0]):
			plt.imsave('predictions/'+pred_folder+'/image_{}.jpg'.format(i), images[i].squeeze(), cmap='gray')
			plt.imsave('predictions/'+pred_folder+'/pred_mask_{}.jpg'.format(i), pred_masks[i].squeeze(), cmap='gray')
			plt.imsave('predictions/'+pred_folder+'/true_mask_{}.jpg'.format(i), true_masks[i].squeeze(), cmap='gray')

		if args.verbose:
			print("## Predictions Saved! ##")