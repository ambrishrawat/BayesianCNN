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


	def train_model(self, options, opti_algo, save_path_key = "models/model_cifar"):
		'''
		Train the model using Stochastic Gradient Descent
	 	(SGD + momentum (how original)).
		'''

		#Bookwork
		batch_size = options['batch_size']
		nb_classes = 10
		nb_epoch = options['epoch']
		data_augmentation = options['data_augmentation']


		#sgd = SGD(lr=opti_param['lr'], decay=opti_param['decay'], momentum=opti_param['momentum'])
		#sgd = Adam(lr=0.01)
		#self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
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

		f = open(save_path_key+'.stats','w')
		f.write('epoch,training_error,validation_error,test_error\n')
		epoch_num = 0
		while epoch_num <= nb_epoch:
			
			self.model.compile(loss='categorical_crossentropy', optimizer=opti_algo)
			self.model.fit(self.X_training, self.Y_training, batch_size=batch_size, nb_epoch=20)
			#def compute_test_error_thresh(self,test_images,test_labels,thresh = 0.5, dropout = False, numiter = 1000):
			training_err = self.compute_test_error_thresh(test_images=self.X_training,test_labels=self.Y_training,thresh=0.0,dropout = False)
			validation_err = self.compute_test_error_thresh(test_images=self.X_validation,test_labels=self.Y_validation,thresh=0.0,dropout = False)
			test_err = self.compute_test_error_thresh(test_images=self.X_test,test_labels=self.Y_test,thresh=0.0,dropout = False)
			epoch_num = epoch_num + 20
			print 'epoch_num: ', epoch_num
			f.write(str(epoch_num)+','+str(training_err)+','+str(validation_err)+','+str(test_err)+'\n')
		json_string = self.model.to_json()
		open(save_path_key+'_arch.json', 'w').write(json_string)
		self.model.save_weights(save_path_key+'_weights.h5')
	

	def load_model(self, tag = 'models/model_cifar'):
		'''
		Load model architecture from a jason file and the corresponding weights from another file
		'''
	  	self.model = Sequential()
		self.model = model_from_json(open(tag+'_arch.json').read())
		self.model.load_weights(tag+'_weights.h5')

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


	def noise_adv_exp(self, noise_img, noise_adv, desired_stats, fp, step = 0.01, num_iter = 1000, stochastic = False):

		'''
		Compute errors - 
		'''
		noise_img = np.array(noise_img)
		noise_img_adv = noise_img.copy()

		labels = K.placeholder(shape=(noise_adv.shape[0],self.nb_classes))

		model_output = self.model.get_output(train=stochastic)
		model_input = self.model.get_input()
 		
		#loss function
		loss = K.mean(categorical_crossentropy(labels,model_output))

		#gradient of loss with respect to input image
		grads = K.gradients(loss,model_input)
		iterate = K.function([model_input,labels], [loss, grads])

		for i in range(0,num_iter+1):
			desired_stats(self,fp, noise_img, noise_img_adv, noise_adv, i)			
			loss_value, grad_value = iterate([noise_img_adv,noise_adv])
			noise_img_adv -= grad_value*step
			print "%16.16f"%np.max(grad_value*(1e6)), 'l2: ', np.linalg.norm(noise_img_adv-noise_img)

		return noise_img_adv

		pass

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

	def compute_test_error_thresh(self,test_images,test_labels,thresh = 0.5, dropout = False, numiter = 1000):
		'''
		Test set evaluation 
		Thresh = 0.0 implies argmax
		Assumes that test label have a binary representation
		'''
		num_img = test_images.shape[0]

		score_mat = []
		if dropout==False:
			score = self.model.predict(test_images)
			#print 'score (shape): ', score.shape
			score_mat.append(score)
		else:
			for i in range(1,numiter):
				score = self.model.predict_stochastic(test_images)
				score_mat.append(score)
		score_mat = np.array(score_mat)
		#print 'score_mat (shape): ', score_mat.shape
		score = np.mean(score_mat,axis=0)
		pred_labels = np.zeros(num_img)
		pred_prob = np.zeros(num_img)
		correct_labels = np.zeros(num_img)

		#TODO: use list comprehension?

		total_correct = 0

		for i in range(num_img):
			correct_labels[i],_ = max(enumerate(test_labels[i]),key=operator.itemgetter(1))
			pred_labels[i],pred_prob[i] = max(enumerate(score[i]),key=operator.itemgetter(1))
			if pred_prob[i] >= thresh and (correct_labels[i] == pred_labels[i]):
				total_correct = total_correct + 1
		error = float(num_img - total_correct)/float(num_img)
		return error


	def class_conditional_stats(self,test_images,test_labels,thresh = 0.5, dropout = False, numiter = 1000):
		'''
		Compute class conditional stats
		'''
		num_img = test_images.shape[0]

		score_mat = []
		if dropout==False:
			score = self.model.predict(test_images)
			#print 'score (shape): ', score.shape
			score_mat.append(score)
		else:
			for i in range(1,numiter):
				score = self.model.predict_stochastic(test_images)
				score_mat.append(score)
		score_mat = np.array(score_mat)
		#print 'score_mat (shape): ', score_mat.shape
		score = np.mean(score_mat,axis=0)
		score_var = np.var(score_mat,axis=0)

		class_pred_prob = np.zeros(num_img)
		class_pred_var = np.zeros(num_img)

		labels = np.zeros(num_img)

		#TODO: use list comprehension?

		total_correct = 0

		for i in range(num_img):
			labels[i],_ = max(enumerate(test_labels[i]),key=operator.itemgetter(1))
			class_pred_prob[i] = score[i][labels[i]]
			class_pred_var[i] = score_var[i][labels[i]]


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

	def noise_exp(self,num_images=1000):		
		'''
		Generate a set of noisy images along with their adversarial labels
		Adverial example for Noisy image
		'''
	
		'''	
		bookwork
		'''
		num_classes = self.nb_classes

		def gen_noise(row=32,col=32,ch=3,mean=0.0,var=0.1):
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(row,col,ch))
			gauss = gauss.reshape(ch,row,col)
			return gauss

		#generate random 100 noisy images, along with their adverarial labels
		gauss = gen_noise()
		noise_test = np.array([gauss])
	
		for _ in range(0,num_images-1):
			gauss = gen_noise()
			noise_test = np.row_stack((noise_test,[gauss]))
	
		#labels 
		labels =  np.random.randint(low=1, high=num_classes,  size=(num_images, 1))
		temp = np.zeros((10,),dtype=np.int)
		temp[labels[0]-1] = 1
		noise_labels = np.array([temp])

		for i in range(1,num_images):
			temp = np.zeros((10,),dtype=np.int)
			temp[labels[i]-1] = 1
			noise_labels = np.row_stack((noise_labels,[temp]))
		
	
		#generate the corresponding adversarial images
		print 'noisy images ', noise_test.shape
		print 'noise_labels ', noise_labels.shape
		print 'X_test ', self.X_test.shape
		print 'Y_test ', self.Y_test.shape

		return noise_test, noise_labels	

	
