#from __future__ import print_function
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
import operator

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

	def save_img(self,img,path='images/img',tag='sample'):
		'''
		Save an image at the specifed path
		'''
		sp.misc.imsave(path+'_'+tag+'.jpg',img[0].T)


	def get_adversarial(self,img,mis_label,stochastic=False):
		'''
		Generate an adversarial example for an image (desired misclassification to label mis_label)
		'''

		labels = K.placeholder(shape=(None,self.nb_classes))

		preds = self.model.get_output(train=stochastic)
		img_placeholder = self.model.get_input()
 
		#loss function
		loss = K.mean(categorical_crossentropy(labels,preds))

		#gradient of loss with respect to input image
		grads = K.gradients(loss,img_placeholder)
		iterate = K.function([img_placeholder,labels], [loss, grads])

		img = np.array(img)
		img_orig = img #(1,3,32,32) instead of (3,32,32) 
		img_adv = img.copy() #(1,3,32,32) instead of (3,32,32)

		temp_label = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) 

		step = 0.01
		for i in range(100):
			loss_value, grads_value = iterate([img_adv,temp_label])
			img_adv -= grads_value*step
			
		return img_adv

	
	def get_img(self,index,train=False):
		'''
		Given index, data (train/test), get the image
		'''		
		img = [self.X_test[index]]
		orig_label_array = np.array(self.Y_test[index])
		if train==True:
			img = [self.X_train[index]]
			orig_label_array = np.array(self.Y_test[index])
			pass
		orig_label,_ = max(enumerate(orig_label_array),key=operator.itemgetter(1))
		return img, orig_label

	def compute_test_error(self,test_images,test_labels):
		'''
		Test set evaluation
		'''
		num_img = test_images.shape[0]
		print 'Number of images in the test set: ', num_img
		score = self.model.predict(test_images)
		print 'score (shape): ', score.shape

		pred_labels = np.zeros(num_img)
		correct_labels = np.zeros(num_img)
		#TODO: use list comprehension?
		for i in range(num_img):
			correct_labels[i],_ = max(enumerate(test_labels[i]),key=operator.itemgetter(1))
			pred_labels[i],_ = max(enumerate(score[i]),key=operator.itemgetter(1))

		total_correct = (correct_labels == pred_labels).sum()
		error = float(num_img - total_correct)/float(num_img)
		print error

	def get_stats(self,img,stochastic=False):
		'''
		Given an image, get the classfication stats from a trained CNN (both with and without-dropout at test time)
		'''
		#get the probability vector (output of the trained CNN)
		img = np.array(img)
		if stochastic==False:
			score = self.model.predict(img) #traditional CNN
			score_mat = np.matrix(score[0])
		else:
			score = self.model.predict_stochastic(img) #with dropout at test time
			score_mat = np.matrix(score[0])
			for i in range(1,100):
				score = self.model.predict_stochastic(img)
				score_mat = np.vstack((score_mat,score[0]))
			#TODO: btch update to make it faster	
		score = np.mean(score_mat,axis=0)
		score = np.squeeze(np.asarray(score))
		#get the predicted label and the predicted score corresponding to that label
		pred_label, pred_score = max(enumerate(score),key=operator.itemgetter(1))		

		return score, pred_label, pred_score, score_mat
		
	@staticmethod
	def print_report(cnn,img):

		#get classification stats from the trained CNN
		c_score, c_pred_label, c_pred_score, c_score_mat = cnn.get_stats(img,stochastic=False)
	
		#get classification stats from the trained Bayesian CNN 
		bc_score, bc_pred_label, bc_pred_score, bc_score_mat = cnn.get_stats(img,stochastic=True)

		#print report
		print 'Traditional CNN'
		print 'Predicted label: ', c_pred_label, ' probability: ', c_pred_score
		print 'Shape (mat): ', c_score_mat.shape, ' Shape (vec): ', c_score.shape	
		print 'Bayesian CNN'		
		print 'Predicted label: ', bc_pred_label, ' probability: ', bc_pred_score
		print 'Shape (mat): ', bc_score_mat.shape, ' Shape (vec): ', bc_score.shape

	def gen_adversarial(self,index,dropout=True):
		'''
		Generate an adversarial example for the index index in cifar10
		'''

		labels = K.placeholder(shape=(None,self.nb_classes))

		#preds = self.ldict['dense_2'].get_output(train=dropout)
		preds = self.model.get_output(train=dropout)
		img_placeholder = self.model.get_input()
 
		#loss function
		loss = K.mean(categorical_crossentropy(labels,preds))

		#gradient of loss with respect to input image
		grads = K.gradients(loss,img_placeholder)
		iterate = K.function([img_placeholder,labels], [loss, grads])


		img_orig = [self.X_test[index]] #(1,3,32,32) instead of (3,32,32) 
		orig_label = self.Y_test[index]
		img_adv = [self.X_test[index]] #(1,3,32,32) instead of (3,32,32)
		temp_label = np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]) 
		step = 0.001
		for i in range(100):
			loss_value, grads_value = iterate([img_adv,temp_label])
			img_adv -= grads_value*step
			#print grads_value
		
		print 'Index: ', index, ' Correct label: ', orig_label	
		sp.misc.imsave('images/img.jpg',img_orig[0].T)
		sp.misc.imsave('images/img_adv.jpg',img_adv[0].T)


		#get classification stats from the trained CNN
		c_score, c_pred_label, c_pred_score, c_score_mat = self.get_stats(img_orig,stochastic=False)
	
		#get classification stats from the trained Bayesian CNN 
		bc_score, bc_pred_label, bc_pred_score, bc_score_mat = self.get_stats(img_orig,stochastic=True)

		#print report
		print 'Traditional CNN'
		print 'Predicted label: ', c_pred_label, ' probability: ', c_pred_score
		print 'Shape (mat): ', c_score_mat.shape, ' Shape (vec): ', c_score.shape, ' Sum (vec): ', np.sum(c_score)
		print 'Bayesian CNN'		
		print 'Predicted label: ', bc_pred_label, ' probability: ', bc_pred_score
		print 'Shape (mat): ', bc_score_mat.shape, ' Shape (vec): ', bc_score.shape, ' Sum (vec): ', np.sum(bc_score)


		#get classification stats from the trained CNN
		c_score, c_pred_label, c_pred_score, c_score_mat = self.get_stats(img_adv,stochastic=False)
	
		#get classification stats from the trained Bayesian CNN 
		bc_score, bc_pred_label, bc_pred_score, bc_score_mat = self.get_stats(img_adv,stochastic=True)

		#print report
		print 'Traditional CNN'
		print 'Predicted label: ', c_pred_label, ' probability: ', c_pred_score
		print 'Shape (mat): ', c_score_mat.shape, ' Shape (vec): ', c_score.shape, ' Sum (vec): ', np.sum(c_score)	
		print 'Bayesian CNN'		
		print 'Predicted label: ', bc_pred_label, ' probability: ', bc_pred_score
		print 'Shape (mat): ', bc_score_mat.shape, ' Shape (vec): ', bc_score.shape, ' Sum (vec): ', np.sum(bc_score)
	
		
