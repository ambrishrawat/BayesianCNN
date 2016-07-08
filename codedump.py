'''
CODE DUMP
'''

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


def exp6():
	'''
	Zero Gradient investigation
	'''
	
	#load model
	cnn = CNN()
	cnn.set_data()
	cnn.load_model()

	
	#hyperparamerters
	num_iter = 20
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
	fpath = 'exp/exp8/log2.txt'
	
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

				cnn.save_img(X_test[j]-X_test_adv[j], path='exp/exp8/gradimg_'+str(j), tag = str(i))
		pass

	#generate random adversarial labels and the corresponding adversarial images for the data set	
	X_test_adv, Y_test_adv = cnn.get_rnd_adv_img(cnn.X_test[1:num_samples+1],cnn.Y_test[1:num_samples+1], \
				desired_stats, fpath, step = step, num_iter = num_iter, stochastic = stochastic)
	pass	

#------------------------------------------------
'''
CNN.py codedump.py
'''
#------------------------------------------------
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

		#TODO: change decision rule

		for i in range(num_img):
			correct_labels[i],_ = max(enumerate(test_labels[i]),key=operator.itemgetter(1))
			pred_labels[i],_ = max(enumerate(score[i]),key=operator.itemgetter(1))
			
		total_correct = (correct_labels == pred_labels).sum()
		error = float(num_img - total_correct)/float(num_img)
		return error
