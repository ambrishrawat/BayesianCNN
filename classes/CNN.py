#from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras.models import model_from_json
from keras.objectives import categorical_crossentropy
from keras.callbacks import Callback
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K 
import operator

class LocalHistory(Callback):
	'''
	Callback for test-error at every epoch
	'''
	def on_epoch_begin(self, epoch, logs={}):
		
		pass
	def on_epoch_end(self, epoch, logs={}):
		if epoch%20==0:
			
			pass
		else:
			pass
		pass

class CNN:
	'''
	CNN Class
	'''
	def __init__(self):
		#TODO
		#Initialise all the class members here (including numpy arrays?)
		#self.X_train = np.array()
		#self.Y_train = np.array()
		#self.X_test = np.array()
		#self.Y_test = np.array()
		
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
		self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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
			input_shape=(img_channels, img_rows, img_cols),name='conv1'))	
		#Layer 1- ReLU activation 
		self.model.add(Activation('relu',name='relu1'))

		#Layer 2 - Convolution2D - 32 filters of size 3x3x32 (32 from the 'depth' of previous Convolution2D Layer)
		#				Output neuron volume size - 32x32x32
		self.model.add(Convolution2D(32, 3, 3,name='conv2'))
		#Layer 2 - ReLU activation
		self.model.add(Activation('relu',name='relu2'))

		#Layer 3 - Pooling Layer (stride = 2, extend 2x2, 75% rejection?) Output neuron volume size - 16x16x32		
		self.model.add(MaxPooling2D(pool_size=(2, 2),name='pool1'))
		self.model.add(Dropout(0.25))
		
		#Layer 4 - Convolution2D - 64 filters of size 3x3x32 (32 from the 'depth' of previous Convolution2D Layer)
		#				Output neuron volume size - 16x16x64
		self.model.add(Convolution2D(64, 3, 3, border_mode='same',name='conv3'))
		#Layer 4 - ReLU activation
		self.model.add(Activation('relu',name='relu3'))

		#Layer 5 - Convolution2D - 64 filters of size 3x3x64 (64 from the 'depth' of previous Convolution2D Layer)
		#				Output neuron volume size - 16x16x64
		self.model.add(Convolution2D(64, 3, 3,name='conv4'))
		#Layer 5 - ReLU activation
		self.model.add(Activation('relu',name='relu4'))

		#Layer 6 - Pooling Layer (stride = 2, extend 2x2, 75% rejection?) Output neuron volume size - 8x8x64		
		self.model.add(MaxPooling2D(pool_size=(2, 2),name='pool'))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		#Layer 7 - Fully connected layer
		self.model.add(Dense(512,name='full1'))
		#Layer 7 - ReLU activation
		self.model.add(Activation('relu',name='relu5'))
		self.model.add(Dropout(0.5))

		#Layer 8 - Fully connected layer
		self.model.add(Dense(self.nb_classes,name='dense2'))
		#Layer 8 - softmax activation
		self.model.add(Activation('softmax',name='softmax1'))

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
		nb_epoch = 420
		data_augmentation = False


		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
		#sgd = Adam(lr=0.01)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
		#self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
		#json_string = self.model.to_json()
		#open(save_path_key+'_arch.json', 'w').write(json_string)
		#self.model.save_weights(save_path_key+'_weights.h5')

		#TODO: compute training, validation and test error plots

		print 'X_train.shape', self.X_train.shape
		print 'Y_train.shape', self.Y_train.shape
		print 'X_test.shape', self.X_test.shape
		print 'X_test.shape', self.Y_test.shape 


		#training set (80% of train) - 40000,3,32,32
		self.X_training = self.X_train[0:40000]
		self.Y_training = self.Y_train[0:40000]
		#validation set (20% of validation), 10000,3,32,32
		self.X_validation = self.X_train[40000:50000]
		self.Y_validation = self.Y_train[40000:50000]
		#test set (100% of test set) - 10000,3,32,32
		self.X_test = self.X_test
		self.Y_test = self.Y_test

		print 'X_training.shape', self.X_training.shape
		print 'Y_training.shape', self.Y_training.shape
		print 'X_validation.shape', self.X_validation.shape
		print 'Y_validation.shape', self.Y_validation.shape
		print 'X_test.shape', self.X_test.shape
		print 'X_test.shape', self.Y_test.shape 

		f = open('training_sgd_2.stats','w')
		f.write('epoch,training_error,validation_error,test_error\n')
		epoch_num = 0
		while epoch_num <= nb_epoch:
			self.model.fit(self.X_training, self.Y_training, batch_size=batch_size, nb_epoch=20)
			training_err = self.compute_test_error(test_images=self.X_training,test_labels=self.Y_training,dropout = False)
			validation_err = self.compute_test_error(test_images=self.X_validation,test_labels=self.Y_validation,dropout = False)
			test_err = self.compute_test_error(test_images=self.X_test,test_labels=self.Y_test,dropout = False)
			epoch_num = epoch_num + 20
			print 'epoch_num: ', epoch_num
			f.write(str(epoch_num)+','+str(training_err)+','+str(validation_err)+','+str(test_err)+'\n')
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
		#TODO: include iterate and grad here, to save time during compilation

	def save_img(self,img,path='images/img',tag='sample'):
		'''
		Save an image at the specifed path
		'''
		sp.misc.imsave(path+'_'+tag+'.jpg',np.rot90(img.T,k=3))


	def get_rnd_adv_img(self,X_test,Y_test, desired_stats, fp, step = 0.01, num_iter = 1000, stochastic = False):

		'''
		genrate adversairal examples for a normal CNN
		Arguments
			- a cnn model 	(cnn)
			- a set of input images	(X_test)
			- a set of labels corresponding to the input images (Y_test)
			- desired_stats (print function, which stats to store)
			- filepath
			- step
			- num_iter
			- stochastic (for BCNN or for CNN)
		Return
			- a set of output images (X_test_adv)
		'''
		X_test = np.array(X_test)
		X_test_adv = X_test.copy()

		#given a Y_test label generate a random adversarial labels. 
		Y_test_adv = self.gen_rnd_adv_label(Y_test)

		labels = K.placeholder(shape=(Y_test.shape[0],self.nb_classes))

		model_output = self.model.get_output(train=stochastic)
		model_input = self.model.get_input()
 		
		#loss function
		loss = K.mean(categorical_crossentropy(labels,model_output))

		#gradient of loss with respect to input image
		grads = K.gradients(loss,model_input)
		iterate = K.function([model_input,labels], [loss, grads])

		for i in range(0,num_iter+1):
			desired_stats(self,fp, X_test, Y_test, X_test_adv, Y_test_adv, i)			
			loss_value, grad_value = iterate([X_test_adv,Y_test_adv])
			X_test_adv -= grad_value*step
			print "%16.16f"%np.max(grad_value*(1e6)), 'l2: ', np.linalg.norm(X_test_adv-X_test)

		return X_test_adv, Y_test_adv
		




		'''
		vanishing gradient investigation
		
		#gradient of loss with respect to input image
		#lay_no = is the ith layer input
		
		loss = K.mean(categorical_crossentropy(labels,model_output))
		lay_no = 10
		layer_i_input = self.ldict['dense_1'].get_input()
		grads = K.gradients(loss,layer_i_input)
		iterate = K.function([layer_i_input,labels], [loss, grads])
		
		get_layer_i = K.function([model_input],[layer_i_input])	
		for i in range(0,num_iter):
			#TODO: change 20 to num_iter
			desired_stats(self,fp, X_test, Y_test, X_test_adv, Y_test_adv, i)			
			linput = get_layer_i([X_test_adv])[0]
			loss_value, grad_value = iterate([linput,Y_test_adv])
			print np.linalg.norm(linput[0])
			X_test_adv -= grad_value*step
			print "%16.16f"%np.max(grad_value*(1)), 'l2: ', np.linalg.norm(X_test_adv-X_test)

		return X_test_adv, Y_test_adv
		'''

	def gen_rnd_adv_label(self,Y_test):
		'''
		Shuffle the labels of the Y_test to get a Y_test_labels
		'''
		Y_test = np.array(Y_test)
		Y_test_adv = Y_test.copy()

		for i in range(Y_test_adv.shape[0]):
			while (Y_test_adv[i]==Y_test[i]).sum() == self.nb_classes:
				temp = Y_test_adv[i]
				np.random.shuffle(temp)
				#TODO: remove the line (only for vanishing gradient experiment)
				#Y_test_adv[i] = np.array([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]]) 
				Y_test_adv[i] = temp
			
		return Y_test_adv	

	
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

	def compute_test_error(self,test_images,test_labels,dropout = False):
		'''
		Test set evaluation (with argmax instead of threshold)
		'''
		num_img = test_images.shape[0]
		#print 'Number of images in the test set: ', num_img
		score_mat = []
		if dropout==False:
			score = self.model.predict(test_images)
			#print 'score (shape): ', score.shape
			score_mat.append(score)
		else:
			for i in range(1,100):
				score = self.model.predict_stochastic(test_images)
				score_mat.append(score)
		score_mat = np.array(score_mat)
		#print 'score_mat (shape): ', score_mat.shape
		score = np.mean(score_mat,axis=0)
		pred_labels = np.zeros(num_img)
		correct_labels = np.zeros(num_img)
		#TODO: use list comprehension?
		for i in range(num_img):
			correct_labels[i],_ = max(enumerate(test_labels[i]),key=operator.itemgetter(1))
			pred_labels[i],_ = max(enumerate(score[i]),key=operator.itemgetter(1))
			
		total_correct = (correct_labels == pred_labels).sum()
		error = float(num_img - total_correct)/float(num_img)
		return error

	def get_stats(self,img,stochastic=False):
		'''
		argument
			image
		return
			score, pred_label, pred_score, score_mat
		'''
		#get the probability vector (output of the trained CNN)
		img = np.array([img])
		score_mat = []
		if stochastic==False:
			score = self.model.predict(img)
			score_mat.append(score)
		else:
			for i in range(1,100):
				score = self.model.predict_stochastic(img)
				score_mat.append(score)
		score_mat = np.array(score_mat)
		score = np.mean(score_mat,axis=0)
		score = score[0]
		#get the predicted label and the predicted score corresponding to that label
		pred_label, pred_score = max(enumerate(score),key=operator.itemgetter(1))		

		return score, pred_label, pred_score, score_mat
		
	@staticmethod
	def print_report(cnn,img):
		'''
		print classificatino stats for a image, given a cnn
		'''
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
		ScracthPad function: experiment for genrating an adversarial example
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
	
		
