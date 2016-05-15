from classes.CNN import CNN
import numpy as np
import operator
def exp1():
	'''
	Generate an adversarial example for an image
	'''

	#pick an image
	index = 2
	#pick from test set
	train = False
	#get the 'original' label (label for a well-trained network
	stochastic=False

	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	
	#get image
	img_orig_label = cnn.get_img(index,train)

	#get classification stats from the trained CNN
	c_score, c_pred_label, c_pred_score = CNN.get_cnn_stats(cnn,img)
	
	#get classification stats from the trained Bayesian CNN 
	bc_score, bc_pred_label, bc_pred_score = CNN.get_bcnn_stats(cnn,img)


if __name__ == "__main__":
	#cnn = CNN()
	#cnn.set_data()
	#cnn.set_model_arch()
	#cnn.train_model()

	#cnn.load_model()
	#cnn.classify_image(2)
	#cnn.classify_image(2)
	#cnn.classify_image(2,stochastic=True)
	#cnn.classify_image(2,stochastic=True)
	#cnn.gen_adversarial(index=2)

	exp1()
