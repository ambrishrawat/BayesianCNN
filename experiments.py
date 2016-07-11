from classes.CNN import CNN
import numpy as np
import operator
import scipy as sp
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop

def exp2():
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model(tag="models/cifar_10_sgd_30612_1")
	err = cnn.compute_test_error_thresh(cnn.X_test[420:450],cnn.Y_test[420:450],thresh = 0.0, dropout = False, numiter = 1000)
	print '\n\n'+str(err)
	



def exp7():
	'''
	SGD training 	
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.set_model_arch()
	options = {'batch_size':32, 'epoch':200, 'data_augmentation':False}

	opti_param = {'lr':0.01, 'decay':1e-6, 'momentum':0.9}
	sgd = SGD(lr=opti_param['lr'], decay=opti_param['decay'], momentum=opti_param['momentum'], nesterov=True)

	cnn.train_model(options=options, opti_algo=sgd, save_path_key = "models/cifar_10_sgd_080716_1")
	

def exp8():
	'''
	RMSprop for training
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.set_model_arch()
	options = {'batch_size':32, 'epoch':200, 'data_augmentation':False}
	
	rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)

	cnn.train_model(options=options, opti_algo=rmsprop, save_path_key = "models/cifar_10_rmsprop_080716_1")


def exp9():
	'''
	Adadelta for training
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.set_model_arch()
	options = {'batch_size':32, 'epoch':200, 'data_augmentation':False}
	
	ada = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

	cnn.train_model(options=options, opti_algo=ada, save_path_key = "models/cifar_10_adadelta_26612_1")


def exp_decision_rule():
	'''
	Different decision rule for different plots 
	'''

	'''
	arguments:
		- number of noisy images
		- 
	'''

	'''
	1.	Make a set of 1000 random noisy-images
	2.	Assign an adversarial label to each of them (without loss of generality, can we assign the same label?)
    3.		
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()

	#generae noise-image data sets
	num_images = 1000
	noise_test, noise_labels = noise_exp(num_images=1000)

	pass




def exp_noise():
	'''

	'''
	date = '_28_6_12'
		

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()

	#load 	
	noise_img, noise_adv_label = cnn.noise_exp()

	#hyperparamerters
	num_iter = 1000
	step = 0.01
	num_samples = 100
	stochastic = False
	fpath = 'exp/exp_noise/log'+date+'.txt'


	#stats file
	f = open(fpath,'w')
	f.write('num_samples: '+ str(num_samples) + '\n'\
		+ 'step: ' + str(step) + '\n'\
		+ 'stochastic: ' + str(stochastic) + '\n')
	
	f.write('iteration,err_cnn_adv,err_bcnn_adv,rms\n')	
	f.close()


	#stats function
	def desired_stats(cnn_m,fp,noise_img, noise_img_adv, adv_label, i):
		if i%10 ==0:		
			f = open(fp,'a')
			ecnn1 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.0, dropout = False)
			ecnn2 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.25, dropout = False)
			ecnn3 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.5, dropout = False)
			ecnn4 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.75, dropout = False)
			
			ebcnn1 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.0, dropout = True, numiter=1000)
			ebcnn2 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.25, dropout = True, numiter=1000)
			ebcnn3 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.5, dropout = True, numiter=1000)
			ebcnn4 = cnn_m.compute_test_error(noise_img, adv_label, thresh = 0.75, dropout = True, numiter=1000)				
			rms = float(np.linalg.norm(noise_img_adv-noise_img))/float(adv_label.shape[0])

			
			print 'Step: ', i, ' Error (CNN): ', ecnn1, ' Error (BayseianCNN): ', ebcnn1,\
				' RMS: ', rms
			f.write(str(i)+','+\
				str(ecnn1)+','+str(ecnn2)+','+\
				str(ecnn3)+','+str(ecnn4)+','+\
				str(ebcnn1)+','+str(ecnn2)+','+\
				str(ebcnn3)+','+str(ecnn4)+','+\
				str(rms)+'\n')
			
			f.close()
			
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	noise_img_adv = cnn.noise_adv_exp(noise_img,noise_adv_label, \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	

def exp_noise_2():
	'''
	Noise l2-
	Fix adversarial class (class label 5)
	Get predictive probablity along with error bars for Bayesian CNN
	Get predictive probablity for the normal CNN	
	'''

	#model specification - date
	date = '_300612'
		
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model(tag='models/cifar_10_sgd_30612_1')

	#load 	
	noise_img, noise_adv_label = cnn.noise_exp(num_images=1000)

	#hyperparamerters
	num_iter = 1000
	step = 0.01
	num_samples = 100
	stochastic = False
	fpath = 'exp/exp_noise/log_mean_var'+date+'.txt'


	#stats file
	f = open(fpath,'w')
	f.write('num_samples: '+ str(num_samples) + '\n'\
		+ 'step: ' + str(step) + '\n'\
		+ 'stochastic: ' + str(stochastic) + '\n')
	
	f.write('iteration,mean_cnn,var_cnn,mean_bcnn,var_bcnn,rms\n')	
	f.close()


	#stats function
	def desired_stats(cnn_m,fp,noise_img, noise_img_adv, adv_label, i):
		if i%10 ==0:		
			f = open(fp,'a')
			cnn_mean, cnn_var = cnn_m.class_conditional_stats(noise_img,adv_label, dropout = False, numiter = 1000)
			bcnn_mean, bcnn_var = cnn_m.class_conditional_stats(noise_img,adv_label, dropout = True, numiter = 1000)				
			rms = float(np.linalg.norm(noise_img_adv-noise_img))/float(adv_label.shape[0])
			print 'Step: ', i, ' Mean-var (CNN): ', cnn_mean, ',', cnn_var,\
			', Mean-var (BCNN): ', bcnn_mean, ',', bcnn_var, \
			' RMS: ', rms
			f.write(str(i)+','+\
				str(cnn_mean)+','+str(cnn_var)+','+\
				str(bcnn_mean)+','+str(bcnn_var)+','+\
				str(rms)+'\n')
			
			f.close()
			
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	noise_img_adv = cnn.noise_adv_exp(noise_img,noise_adv_label, \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass


if __name__ == "__main__":
	exp7()
	pass
