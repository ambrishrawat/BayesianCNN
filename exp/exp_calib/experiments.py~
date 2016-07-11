from classes.CNN import CNN
import numpy as np
import operator
import scipy as sp
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop



def exp_calib():
	'''
	Calibration
	'''
		
	#model specification - date
	date = '_300612'
		
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model(tag='models/cifar_10_sgd_30612.json')

	c_label = 5
	'''
	Strep1: get the class conditional probabilities for a fixed class label (in this case label=5)
	'''
	#generate a fixed label test set (only label)
	_, Y_test = noise_exp(self,num_images=10000, options = {'random':False, 'fixed_label':c_label})
	
	#compute class probabilities for the whole test set
	class_pred_prob, class_pred_var = cnn.fix_class_conditional_stats(cnn.X_test, Y_test, dropout = True, numiter = 1000, ret_mean = False)
	
	'''
	Step 1: Bin the probabilities of classification. 
	Fixing bin size might result in noisy estimation. Instead do it dynamically. Fix number of samples and fing the probablity bin accordingly
	Step 2: 	
	'''

	bins = []
	i = 0.0
	while i < 1.0:
		imgs_denom = np.where(class_pred_prob>i and class_pred_prob<i+0.01)[0]
		denom = imgs_denom.size
		Y_test_temp = cnn.Y_test[imgs_denom]
		imgs_numer = np.where(Y_test_temp==c_label)[0]
		numer = imgs_numer.size
		calib_prob = float(numer)/float(denom)
		i = i + 0.01

	pass	

if __name__ == "__main__":
	exp7()
	pass
