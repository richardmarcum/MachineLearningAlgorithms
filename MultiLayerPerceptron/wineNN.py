#!/bin/python3

from scipy.io import loadmat
import numpy as np
from NeuralNetwork import *
from sklearn.preprocessing import scale






if __name__ == "__main__":

	Wine = loadmat('wine.mat')
	Wine = Wine['wine']
	Data = Wine[:,1:]
	Labels = Wine[:,:1]

	#this makes the data have mean 0 and std 1
	Data = scale(Data)



	NumPoints = len(Data)

	rp = np.random.permutation(NumPoints)
	Data=Data[rp,:]
	Labels=Labels[rp,:]

	TrainingSize = int(0.9 * NumPoints)


	XTrain = Data[:TrainingSize]
	YTrain = Labels[:TrainingSize]
	XTest = Data[TrainingSize:]
	YTest = Labels[TrainingSize:]

	Dimensions = (13,3)
	layerActivations = (
						ActivationFunction.Sigmoid, 
						#ActivationFunction.Sigmoid,
						#ActivationFunction.Sigmoid,
						#ActivationFunction.Softmax
						)

	MLP = MultiLayerPerceptron(Dimensions, layerActivations, 
							   learningRate = 0.9, momentum = 0, regularizer=0)

	print(MLP.weights)

	MLP.TrainEpoch(XTrain, YTrain, 5, BatchTrain=True)

	pred = MLP.TestData(XTest)

	correct = 0
	for i in range(len(pred)):
		if pred[i] == YTest[i]:
			correct = correct+1

	print("The MLP got {} correct".format(correct))

	# for i in range(30):
	#      print(pred[i],YTest[i][0])
