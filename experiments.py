from classes.CNN import CNN
import numpy as np
import operator
import scipy as sp

def exp2():
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	
	#get image
	#pick an image
	index = 42
	#pick from test set
	train = False

	cnn.gen_adversarial(index,True)



def exp7():
	'''
	Retrain model. Save errors after every epoch
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.set_model_arch()
	cnn.train_model(save_path_key = "models/model_cifar_2_adam")
	
	
def exp13():
	'''

	'''

	
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
	fpath = 'exp/exp13/log.txt'


	#stats file
	f = open(fpath,'w')
	f.write('num_samples: '+ str(num_samples) + '\n'\
		+ 'step: ' + str(step) + '\n'\
		+ 'stochastic: ' + str(stochastic) + '\n')
	
	f.write('iteration,err_cnn,err_bcnn,err_cnn_adv,err_bcnn_adv,rms\n')	
	f.close()


	#stats function
	def desired_stats(cnn_m,fp,noise_img, noise_img_adv, adv_label, i):
		if i%10 ==0:		
			f = open(fp,'a')
			ecnn = cnn_m.compute_test_error(noise_img, adv_label, dropout = False)
			ebcnn = cnn_m.compute_test_error(noise_img, adv_label, dropout = True)				
			rms = float(np.linalg.norm(noise_img_adv-noise_img))/float(adv_label.shape[0])

			
			print 'Step: ', i, ' Error (CNN): ', ecnn, ' Error (BayseianCNN): ', ebcnn,\
				' RMS: ', rms
			f.write(str(i)+','+str(ecnn)+','+str(ebcnn)+','+str(rms)+'\n')
			
			f.close()
			
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	noise_img_adv = cnn.noise_adv_exp(noise_img,noise_adv_label, \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	

if __name__ == "__main__":
	exp13()
	pass
