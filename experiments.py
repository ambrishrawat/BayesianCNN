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
	img_adv = cnn.get_adversarial(img,mis_label, desired_stats)

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
	

	#hyperparamerters
	num_iter = 10000
	step = 0.01
	num_samples = 100
	stochastic = False
	fpath = 'stats/stats_18_5_1600.txt'


	#stats file
	f = open(fpath,'w')
	f.write('num_samples: '+ str(num_samples) + '\n'\
		+ 'step: ' + str(step) + '\n'\
		+ 'stochastic: ' + str(stochastic) + '\n')
	
	f.write('iteration,err_cnn,err_bcnn,err_cnn_adv,err_bcnn_adv,rms\n')	
	f.close()


	#stats function
	def desired_stats(cnn_m,fp, X_test, Y_test, X_test_adv, Y_test_adv, i):
		if i%10 ==0:		
			f = open(fp,'a')
			ecnn, ecnn_adv = cnn_m.compute_test_error(X_test_adv,Y_test, Y_test_adv, dropout = False)
			ebcnn, ebcnn_adv = cnn_m.compute_test_error(X_test_adv,Y_test, Y_test_adv, dropout = True)				
			rms = float(np.linalg.norm(X_test_adv-X_test))/float(Y_test.shape[0])
			print 'Step: ', i, ' Error (CNN): ', ecnn, ' Error (BayseianCNN): ', ebcnn,\
				' Error adv (CNN): ', ecnn_adv, ' Error adv (BayseianCNN): ', ebcnn_adv,\
				' RMS: ', rms
			f.write(str(i)+','+str(ecnn)+','+str(ebcnn)+','+\
				str(ecnn_adv)+','+str(ebcnn_adv)+','+str(rms)+'\n')
			f.close()
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	X_test_adv, Y_test_adv = cnn.get_rnd_adv_img(cnn.X_test[0:num_samples],cnn.Y_test[0:num_samples], \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	

def exp5():
	'''
	Generate a few sample images and visualise their output adversarial image 
	'''
	
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	

	#hyperparamerters
	num_iter = 100000
	step = 0.01
	num_samples = 1
	stochastic = False
	fpath = 'img_stats/log.txt'
	
	'''
	#10 images belonging to 10 classes and save them
	indices = [0, 1, 433, 4, 225, 37, 557, 42, 21, 357]
	#indices = [0]
	num_samples = len(indices)
	labels = cnn.labels
	test_images = []
	test_labels = []
	for i in range(len(indices)):
		test_images.append(cnn.X_test[indices[i]])
		test_labels.append(cnn.Y_test[indices[i]])
		label_index,_ = max(enumerate(test_labels[i]),key=operator.itemgetter(1))
		print 'image: ', i, ' label_index: ', label_index, 'label: ', labels[label_index] 
		cnn.save_img(test_images[i],path='img_stats/img',tag=str(i))
	
	test_images = np.array(test_images)
	test_labels = np.array(test_labels)

	'''

	#stats file
	f = open(fpath,'w')
	f.write('num_samples: '+ str(num_samples) + '\n'\
		+ 'step: ' + str(0.01) + '\n'\
		+ 'stochastic: ' + str(False) + '\n')
	
	f.write('image file,iteration,original label, intended adv label, cnn_pred_label, bcnn_adv_label, pred_cnn_score, pred_bcnn_score, rms\n')
	f.close()
	#stats function
	def desired_stats(cnn_m,fp, X_test, Y_test, X_test_adv, Y_test_adv, i):
		if i%1000==0:
			print 'Iteration ', i
		if i in [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:		
			#take every image in the X_test_adv and classify it. 
			#Save the label in the log and save the image
			for j in range(X_test.shape[0]):
				_, pred_cnn_label, pred_cnn_score, _ = cnn.get_stats(X_test_adv[j],stochastic=False)
				_, pred_bcnn_label, pred_bcnn_score, _ = cnn.get_stats(X_test_adv[j],stochastic=True)
				orig_label,_ = max(enumerate(Y_test[j]),key=operator.itemgetter(1))
				int_adv_label, _ = max(enumerate(Y_test_adv[j]),key=operator.itemgetter(1))
				f = open(fp,'a')
				rms = float(np.linalg.norm(X_test_adv[j]-X_test[j]))
				f.write('img_'+str(j)+'_'+str(i)+','\
					+cnn.labels[orig_label] + ','+ cnn.labels[int_adv_label] + ',' \
					+cnn.labels[pred_cnn_label] + ',' + cnn.labels[pred_bcnn_label] + ',' \
					+str(pred_cnn_score) + ',' + str(pred_bcnn_score) + ','+ str(rms)+'\n')
				f.close()

				cnn.save_img(X_test[j], path='img_stats/img_'+str(j), tag = str(i))
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	X_test_adv, Y_test_adv = cnn.get_rnd_adv_img(cnn.X_test[1:num_samples+1],cnn.Y_test[1:num_samples+1], \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	
	
def exp5():
	'''
	Generate a few sample images and visualise their output adversarial image 
	'''
	
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()
	

	#hyperparamerters
	num_iter = 100000
	step = 0.01
	num_samples = 1
	stochastic = False
	fpath = 'img_stats/log.txt'

	#stats file
	f = open(fpath,'w')
	f.write('num_samples: '+ str(num_samples) + '\n'\
		+ 'step: ' + str(0.01) + '\n'\
		+ 'stochastic: ' + str(False) + '\n')
	
	f.write('image file,iteration,original label, intended adv label, cnn_pred_label, bcnn_adv_label, pred_cnn_score, pred_bcnn_score, rms\n')
	f.close()
	#stats function
	def desired_stats(cnn_m,fp, X_test, Y_test, X_test_adv, Y_test_adv, i):
		if i%1000==0:
			print 'Iteration ', i
		if i in [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:		
			#take every image in the X_test_adv and classify it. 
			#Save the label in the log and save the image
			for j in range(X_test.shape[0]):
				_, pred_cnn_label, pred_cnn_score, _ = cnn.get_stats(X_test_adv[j],stochastic=False)
				_, pred_bcnn_label, pred_bcnn_score, _ = cnn.get_stats(X_test_adv[j],stochastic=True)
				orig_label,_ = max(enumerate(Y_test[j]),key=operator.itemgetter(1))
				int_adv_label, _ = max(enumerate(Y_test_adv[j]),key=operator.itemgetter(1))
				f = open(fp,'a')
				rms = float(np.linalg.norm(X_test_adv[j]-X_test[j]))
				f.write('img_'+str(j)+'_'+str(i)+','\
					+cnn.labels[orig_label] + ','+ cnn.labels[int_adv_label] + ',' \
					+cnn.labels[pred_cnn_label] + ',' + cnn.labels[pred_bcnn_label] + ',' \
					+str(pred_cnn_score) + ',' + str(pred_bcnn_score) + ','+ str(rms)+'\n')
				f.close()

				cnn.save_img(X_test[j], path='img_stats/img_'+str(j), tag = str(i))
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	X_test_adv, Y_test_adv = cnn.get_rnd_adv_img(cnn.X_test[1:num_samples+1],cnn.Y_test[1:num_samples+1], \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	
	
if __name__ == "__main__":
	
	#exp1()
	#exp2()
	#exp4()
	exp5()
