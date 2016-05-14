from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
from keras.objectives import categorical_crossentropy
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K 

class CNN:
	'''
	CNN Class
	'''
	def __init__(self):
		#TODO
		#Initialise all the class members here (including numpy arrays?)
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
	

		#Layer 1 - Convolution2D - 32 filters of size 3x3ximg_channels, Output neuron volume size - 32x32x32
		self.model.add(Convolution2D(32, 3, 3, border_mode='same',\
			input_shape=(img_channels, img_rows, img_cols),name='conv1_1'))	
		#Layer 1- ReLU activation 
		self.model.add(Activation('relu'))

		#Layer 2 - Convolution2D - 32 filters of size 3x3x32 (32 from the 'depth' of previous Convolution2D Layer)
		#				Output neuron volume size - 32x32x32
		self.model.add(Convolution2D(32, 3, 3,name='conv1_2'))
		#Layer 2 - ReLU activation
		self.model.add(Activation('relu'))

		#Layer 3 - Pooling Layer (stride = 2, extend 2x2, 75% rejection?) Output neuron volume size - 16x16x32		
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))
		
		#Layer 4 - Convolution2D - 64 filters of size 3x3x32 (32 from the 'depth' of previous Convolution2D Layer)
		#				Output neuron volume size - 16x16x64
		self.model.add(Convolution2D(64, 3, 3, border_mode='same',name='conv2_1'))
		#Layer 4 - ReLU activation
		self.model.add(Activation('relu'))

		#Layer 5 - Convolution2D - 64 filters of size 3x3x64 (64 from the 'depth' of previous Convolution2D Layer)
		#				Output neuron volume size - 16x16x64
		self.model.add(Convolution2D(64, 3, 3,name='conv2_2'))
		#Layer 5 - ReLU activation
		self.model.add(Activation('relu'))

		#Layer 6 - Pooling Layer (stride = 2, extend 2x2, 75% rejection?) Output neuron volume size - 8x8x64		
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		#Layer 7 - Fully connected layer
		self.model.add(Dense(512,name='dense_1'))
		#Layer 7 - ReLU activation
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))

		#Layer 8 - Fully connected layer
		self.model.add(Dense(self.nb_classes,name='dense_2'))
		#Layer 8 - softmax activation
		self.model.add(Activation('softmax'))

		#intiliase a dictionary of layers
		self.ldict = dict([(layer.name, layer) for layer in self.model.layers])
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

	

	def load_model(self):
		'''
		Load model architecture from a jason file and the corresponding weights from another file
		'''
	  	self.model = Sequential()
		self.model = model_from_json(open('models/model_cifar_arch.json').read())
		self.model.load_weights('models/model_cifar_weights.h5')

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
		
		#intiliase a dictionary of layers
		self.ldict = dict([(layer.name, layer) for layer in self.model.layers])

	def save_img(self,index,path='images/img'):
		'''
		Save an image at the specifed path
		'''
		img = np.array(self.X_train[index])
		sp.misc.imsave(path+'_'+str(index)+'.jpg',img[0].T)

	def classify_image(self,index,train=False):
		'''
		Given a trained network classify an image
		'''
		if train==True:
			pass
		else:
			pass
		#self.model.predict
		pass

	def gen_adversarial(self,index,dropout=True):
		'''
		Generate an adversarial example for the index index in cifar10
		'''

		labels = K.placeholder(shape=(None,self.nb_classes))

		preds = self.ldict['dense_2'].get_output(train=dropout)
		img_placeholder = self.model.get_input()
 
		#loss function
		loss = K.mean(categorical_crossentropy(labels,preds))

		#gradient of loss with respect to input image
		grads = K.gradients(loss,img_placeholder)
		iterate = K.function([img_placeholder,labels], [loss, grads])


		img_orig = [self.X_train[index]] #(1,3,32,32) instead of (3,32,32) 
		orig_label = self.Y_train[index]
		img_adv = [self.X_train[index]] #(1,3,32,32) instead of (3,32,32)
		temp_label = np.array([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]]) 
		step = 0.01
		for i in range(100):
			loss_value, grads_value = iterate([img_adv,temp_label])
			img_adv += grads_value*step
			
		sp.misc.imsave('images/img.jpg',img_orig[0].T)
		sp.misc.imsave('images/img_adv.jpg',img_adv[0].T)


	
