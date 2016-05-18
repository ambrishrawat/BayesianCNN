from classes.CNN import CNN
import numpy as np
import operator
def exp1():
	'''
	Get some stats for a model and an image (sanity check)
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	
	#get image
	#pick an image
	index = 2
	#pick from test set
	train = False
	img, orig_label = cnn.get_img(index,train)

	#get classification stats from the trained CNN
	c_score, c_pred_label, c_pred_score = cnn.get_stats(img,stochastic=False)
	
	#get classification stats from the trained Bayesian CNN 
	bc_score, bc_pred_label, bc_pred_score = cnn.get_stats(img,stochastic=True)


	#print report
	print 'Image details:'
	if train==True:
		print 'Training set, index: ', index, ' original label: ',orig_label
	else:
		print 'Test set, index: ', index, 'original label: ',orig_label
	
	print 'Prediction from trained CNN'
	print 'Predicted label: ', c_pred_label, ' probability: ', c_pred_score
	print 'Prediction from trained CNN with droput at test time'
	print 'Predicted label: ', bc_pred_label, ' probability: ', bc_pred_score


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


def exp3():
	'''
	Get some stats for a model and an image (sanity check)
	'''

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	
	#get image
	#pick an image
	index = 2
	#pick from test set
	train = False
	img, orig_label = cnn.get_img(index,train)
	CNN.print_report(cnn,img)	

	#misclassification label
	mis_label = 2
	
	#get adversarial image corresponding to img and intended misclassification to label mis_label
	img_adv = cnn.get_adversarial(img,mis_label)

	#print report
	CNN.print_report(cnn,img_adv)


def exp4():
	'''
	Get test error as a function of gradient steps (0-10000)
	for an adversarial examples for a normal CNN 
	'''
	
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	
	#generate random adversarial labels and the corresponding adversarial images for the data set	
	X_test_adv, Y_test_adv, err_cnn, err_bcnn, rms = cnn.get_rnd_adv_img(cnn.X_test[0:100],cnn.Y_test[0:100])
	print 'X_test_adv.shape: ', X_test_adv.shape, ' Y_test_adv.shape: ', Y_test_adv.shape
	print 'err_cnn.shape: ', err_cnn.shape, ' err_bcnn.shape: ', err_bcnn.shape, 'rms.shape: ', rms.shape
	np.save('error_cnn.npy',err_cnn)
	np.save('error_bcnn.npy',err_bcnn)
	np.save('rms.npy',rms)
	pass	
	
if __name__ == "__main__":
	
	#exp1()
	#exp2()
	exp4()
