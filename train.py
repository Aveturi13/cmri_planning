""" Train neural networks architectures using Keras"""

#Import packages
import sys
import numpy as np
import os
import argparse
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data_processing import semseg, paramest
from metrics.dice import dice_score
from networks import unet, alexnet

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.metrics import accuracy
from tensorflow.keras.metrics import MeanIoU
from keras import backend as K

#Get current time
now = time.localtime()
current_time = time.strftime("%H:%M:%S", now)

models = ['unet', 'alexnet']
#model_choices = ['unet', 'alexnet']
#losses = ['BCE', 'MSE']

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Models to train', choices=models)
	parser.add_argument('--epochs', help='Number of epochs to train', type=int, default=30)
	parser.add_argument('--batch_size', help='Batch size', default=32, type=int)
	parser.add_argument('--lr', default=1e-02, help='Learning rate', type=float)
	parser.add_argument('--img_dir', help='Filepath to images dataset')
	parser.add_argument('--params', help='Path to csv file containing 2Ch line parameters')
	parser.add_argument('--split', help='Proportion of data to reserve for testing', default=0.25)
	parser.add_argument('--resize_dim', default=(160, 160), help='Dimensions to resize the image')
	#parser.add_argument('--model', help="Name of model to train on", choices=model_choices)
	parser.add_argument('--callbacks', action='store_true', help="Set up model callbacks")
	#parser.add_argument('--loss', help="Loss function to optimize model", choices=losses)
	#parser.add_argument('--model_log_dir', help="Filepath to save model logs", type=str)
	parser.add_argument('--model_save_dir', help="Filepath to save trained models", type=str)
	parser.add_argument('--batchnorm', action='store_true', help='Adds batch normalization')
	parser.add_argument('--dropout', help='Amount of dropout regularlization. Must be a value in [0, 1).', default=0.0)
	parser.add_argument('--verbose', action='store_true', help='Verbose')

	args= parser.parse_args()

	#Specify a directory if not provided
	if not args.img_dir or not args.params:
		print('Missing the Image / Params directory!')
		sys.exit(1)

	# Check if gpu availible
	physical_devices = tf.config.list_physical_devices('GPU')
	print("Number of GPUs: ", len(physical_devices))

	#Setup model configuration dictionary
	model_config = {'model': args.model,
	                'epochs': args.epochs,
	 				'batch_size': args.batch_size,
	 				'lr': args.lr,
	 				'img_dir': args.img_dir,
	 				'params': args.params,
	 				'split':args.split,
	 				'resize_dim':args.resize_dim,
	 				#'model': args.model,
	 				'callbacks': args.callbacks,
	 				#'loss': args.loss,
	 				#'model_log_dir': args.model_log_dir,
	 				'model_save_dir': args.model_save_dir,
	 				'batchnorm': args.batchnorm,
	 				'dropout': args.dropout,
	 				'verbose': args.verbose}

	#Create model
	if args.model == 'unet':
		if args.verbose:
			print("\n{} Creating Dataset...".format(current_time))

		data = semseg.Dataset(model_config['img_dir'], model_config['params'])

		if args.verbose:
			print("Number of patients : " + str(data.len))

		if args.verbose:
			print("{} Splitting into Train/Test sets...".format(current_time))

		train_X, train_y, val_X, val_y = data.split(shuffle=True, split=model_config['split'])

		if args.verbose:
			print("{} Previewing an Image...".format(current_time))

		data.generate_preview(train_X[0], train_y[0], model_config['resize_dim'])

		if args.verbose:
			print("{} Creating Data Generator...".format(current_time))

		train_generator = data.data_generator(train_X, train_y, resize_dim=model_config['resize_dim'], batch_size=model_config['batch_size'])
		validation_data = data.preprocess_data(val_X, val_y)

		if args.verbose:
			print("{} Setting up Model...\n".format(current_time))

		model = unet.unet(batchnorm=model_config['batchnorm'], dropout=model_config['dropout'])

		#Print a model summary
		if args.verbose:
			print("Model Summary:")
			print(model.summary())

		if args.verbose:
			print("\n{} Compiling Model...".format(current_time))

		optimizer = Adam(lr=0.01)
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[MeanIoU(num_classes=2)])

	if args.model == 'alexnet':
		#Add code on data generators!
		train_generator = None
		validation_data = None
		model = alexnet(batchnorm=model_config['batchnorm'], dropout=model_config['dropout'])
		model.compile(optimizer=Adam(), loss='mean_square_error', metrics=['accuracy'])

	if not args.model:
		print('Model must be specified.')
		sys.exit(1)

	#Create model
	#if args.model == 'unet':
	#	model = unet(batchnorm=model_config['batchnorm'], dropout=model_config['dropout'])
	#if args.model == 'alexnet':
	#	model = alexnet(batchnorm=model_config['batchnorm'], dropout=model_config['dropout'])
	#else:
	#	print('No model selected!')
	#	sys.exit(1)

	#Losses
	#if args.loss == 'BCE':
	#	loss = 'binary_cross_entropy'
	#if args.loss == 'MSE':
	#	loss = 'mean_square_error'
	#else:
	#	print('Please provide a loss function!')
	#	sys.exit(1)

	#Compile model
	#model.compile(optimizer=Adam(lr=args.lr), loss=loss, metrics=[])

	def filename():
		""" Creates a human readable filename"""
		return '{}-bn{}-{}dp-{}e-{}bs-{}lr.h5'.format(model_config['model'],
												  model_config['batchnorm'],
												  model_config['dropout'],
												  model_config['epochs'],
												  model_config['batch_size'],
												  model_config['lr'])

	#Build callbacks
	def build_callbacks(checkpoints=True, tensorboard=True):
		""" Set model callbacks here """

		callbacks = []

		if checkpoints:

			#Create checkpoints directory
			if not os.path.exists('checkpoints'):
				os.mkdir('checkpoints')

			#Monitor different metrics based on type of model
			if model_config['model'] == 'unet':
				checkpoint = ModelCheckpoint(filepath = 'checkpoints/'+filename(),
											monitor= 'val_dice_score',
											verbose=1,
											save_best_only=True,
											mode='max')

			if model_config['model'] == 'alexnet':
				checkpoint = ModelCheckpoint(filepath = 'checkpoints/'+filename(),
											monitor= 'val_accuracy',
											verbose=1,
											save_best_only=True,
											mode='max')

			callbacks.append(checkpoint)

		if tensorboard:
			#Create logs directory
			if not os.path.exists('logs/'):
				os.mkdir('logs/')

			log_dir = os.path.join('logs/', filename()[:-3])
			tensorboard_callback = TensorBoard(log_dir=log_dir)
			callbacks.append(tensorboard_callback)

		return callbacks

	if model_config['callbacks']:
		if args.verbose:
			print("{} Setting up Callbacks...".format(current_time))
		callbacks = build_callbacks()
	else:
		callbacks = None

	#Train model
	if args.verbose:
		print("{} Model Training...\n".format(current_time))

	train_steps = len(train_X) // model_config['batch_size']

	model.fit(x=train_generator,
			  epochs= model_config['epochs'],
			  verbose= model_config['verbose'],
			  callbacks=callbacks,
			  validation_data=validation_data,
			  steps_per_epoch=train_steps)

	if args.verbose:
		print("{} Training Complete\n".format(current_time))

	#Save model
	if args.model_save_dir:

		if args.verbose:
			print("{} Saving Model...".format(current_time))

		#Create the directory, if it doesn't exist
		if not os.path.exists(model_config['model_save_dir']):
			os.mkdir(model_config['model_save_dir'])

		model.save(os.path.join(model_config['model_save_dir'], filename()))

		if args.verbose:
			print("{} Model Saved!".format(current_time))
