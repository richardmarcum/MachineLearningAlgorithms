#!/bin/python3

from scipy.io import loadmat
import numpy as np
from NeuralNetwork import *
from sklearn.preprocessing import scale

def CleanSimeonLabels(Labels):
	labelList = []
	for x in Labels:
		digit = np.argmax(x)
		labelList.append(digit)
	labelList = np.array(labelList)
	labelList = labelList.reshape(labelList.shape[0],1)

	return labelList





if __name__ == "__main__":

	Semeion = loadmat('semeion.mat')
	Semeion = Semeion['semeion']
	Data = Semeion[:,:256]
	origLabels = Semeion[:,256:]

	Labels = CleanSimeonLabels(origLabels)

	#this makes the data have mean 0 and std 1
	#Data = scale(Data)



	NumPoints = len(Data)

	rp = np.random.permutation(NumPoints)
	Data=Data[rp,:]
	Labels=Labels[rp]

	TrainingSize = int(0.9 * NumPoints)


	XTrain = Data[:TrainingSize]
	YTrain = Labels[:TrainingSize]
	XTest = Data[TrainingSize:]
	YTest = Labels[TrainingSize:]

	Dimensions = (256,20,10)
	layerActivations = (
						ActivationFunction.Sigmoid, 
						ActivationFunction.Softmax,
						#ActivationFunction.Linear,
						ActivationFunction.Softmax
						)

	MLP = MultiLayerPerceptron(Dimensions, layerActivations, 
							   learningRate = 0.7, momentum = 0.5, regularizer=0)

	print(MLP.weights)

	MLP.TrainEpoch(XTrain, YTrain, 20, BatchTrain=False)

	pred = MLP.TestData(XTest)

	correct = 0
	for i in range(len(pred)):
		if pred[i] == YTest[i]:
			correct = correct+1

	print("The MLP got {} correct".format(correct))

	# for i in range(30):
	#      print(pred[i],YTest[i][0])
