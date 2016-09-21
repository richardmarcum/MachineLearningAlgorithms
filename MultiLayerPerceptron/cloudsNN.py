#!/bin/python3

from scipy.io import loadmat
import numpy as np
from NeuralNetwork import *
from sklearn.preprocessing import scale






if __name__ == "__main__":

	Clouds = loadmat('threeclouds.mat')
	Clouds = Clouds['threeclouds']

	Data = Clouds[:,[1,2]]
	Labels = Clouds[:,[0]]
	NumPoints = len(Data)

	rp = np.random.permutation(NumPoints)
	Data=Data[rp,:]
	Labels=Labels[rp,:]

	TrainingSize = int(0.9 * NumPoints)


	XTrain = Data[:TrainingSize]
	YTrain = Labels[:TrainingSize]
	XTest = Data[TrainingSize:]
	YTest = Labels[TrainingSize:]

	Dimensions = (2,4,3)
	layerActivations = (
						ActivationFunction.Sigmoid, 
						ActivationFunction.Sigmoid,
						#ActivationFunction.Linear,
						#ActivationFunction.Softmax
						)

	MLP = MultiLayerPerceptron(Dimensions, layerActivations, 
							   learningRate = 0.2, momentum = .75, regularizer=1)

	print(MLP.weights)

	MLP.TrainEpoch(XTrain, YTrain, 20, BatchTrain=True)

	pred = MLP.TestData(XTest)

	correct = 0
	for i in range(len(pred)):
		if pred[i] == YTest[i]:
			correct = correct+1

	print("The MLP got {} correct".format(correct))

	# for i in range(30):
	#      print(pred[i],YTest[i][0])
