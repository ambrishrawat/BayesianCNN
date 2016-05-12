from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K 

class CNN:
	'''
	CNN Class
	'''
	def __init__(self):
		pass

	def set_data(self):
		'''
		get cifar10 data
		'''
		(self.X_train, y_train), (self.X_test, y_test) = cifar10.load_data()	

		#Bookwork
		self.nb_classes = 10

		# convert class vectors to binary class matrices
		self.Y_train = np_utils.to_categorical(y_train, self.nb_classes)
		self.Y_test = np_utils.to_categorical(y_test, self.nb_classes)
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train /= 255
		self.X_test /= 255


	def set_model_arch(self):
		'''
		Deine the architecture for the CNN
		'''
		self.model = Sequential()

		

		# input image dimensions
		img_rows, img_cols = 32, 32
		# the CIFAR10 images are RGB
		img_channels = 3
	
		#First set of layers	
		self.model.add(Convolution2D(32, 3, 3, border_mode='same',\
			input_shape=(img_channels, img_rows, img_cols),name='conv1_1'))	
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(32, 3, 3,name='conv1_2'))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		#Second set of layers
		self.model.add(Convolution2D(64, 3, 3, border_mode='same',name='conv2_1'))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, 3, 3,name='conv2_2'))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		#Third set of layers
		self.model.add(Flatten())
		self.model.add(Dense(512,name='dense_1'))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(self.nb_classes,name='dense_2'))
		self.model.add(Activation('softmax'))
		pass


	def train_model(self,save_path_key = "models/model_cifar"):
		'''
		Train the model using Stochastic Gradient Descent
	 	(SGD + momentum (how original)).
		'''

		#Bookwork
		batch_size = 32
		nb_classes = 10
		nb_epoch = 200
		data_augmentation = False


		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
		self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
		json_string = self.model.to_json()
		open(save_path_key+'_arch.json', 'w').write(json_string)
		self.model.save_weights(save_path_key+'_weights.h5')

	def get_layer_dict(self):
		'''
		get the dictionary of layers
		'''
		layer_dict = dict([(layer.name, layer) for layer in model.layers])
		return layer_dict

	def load_model(self):
		'''
		Load model architecture from a jason file and the corresponding weights from another file
		'''
	  	self.model = Sequential()
		self.model = model_from_json(open('models/model_200_arch.json').read())
		self.model.load_weights('models/model_200_weights.h5')

	def classify_image(self,img):
		'''
		Given a trained network classify an image
		'''		
		pass

	
		
