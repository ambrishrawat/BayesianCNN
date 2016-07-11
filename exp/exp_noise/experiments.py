from classes.CNN import CNN
import numpy as np
import operator
import scipy as sp
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop



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
	cnn.load_model(tag='models/cifar_10_sgd_30612.json')

	#load 	
	noise_img, noise_adv_label = cnn.noise_exp(num_images=1000, options = {'random':False, 'fixed_label':5)

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

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	noise_img_adv = cnn.noise_adv_exp(noise_img,noise_adv_label, \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	

if __name__ == "__main__":
	exp7()
	pass
