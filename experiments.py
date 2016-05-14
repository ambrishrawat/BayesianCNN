from classes.CNN import CNN

if __name__ == "__main__":
	cnn = CNN()
	cnn.set_data()
	#cnn.set_model_arch()
	#cnn.train_model()

	cnn.load_model()
	cnn.classify_image(2)
	cnn.classify_image(2)
	cnn.classify_image(2,stochastic=True)
	cnn.classify_image(2,stochastic=True)
	#cnn.gen_adversarial(index=2)
