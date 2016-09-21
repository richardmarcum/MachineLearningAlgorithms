#!/bin/bash

import numpy as np
from NeuralNetwork import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale

# def CleanAndShuffleIrisData(iris Percent=0.8):

# 	rp = np.random.permutation(Inputs.shape[0])
# 	X = iris.data[rp,:]
# 	Y = iris.data[rp]
# 	CleanedLabels = OneHotEncode(Y)

# 	Num = X.shape[0] * Percent

# 	XTrain = X[:Num]
# 	XTest = X[Num:]
# 	YTrain = CleanedLabels[:Num]
# 	YTest = CleanedLabels[Num:]


if __name__ == "__main__":



	# load, shuffle, data 
	iris = load_iris()
	Data = scale(iris.data)
	rp = np.random.permutation(iris.data.shape[0])
	X = Data[rp,:]
	Y = iris.target[rp]
	XTrain = X[:120]
	XTest = X[120:]
	YTrain = Y[:120]
	YTest = Y[120:]

	# Dimensions = (4, 3)
	# layerActivations = (ActivationFunction.Sigmoid, None)
	
	Dimensions = (4, 3, 3)
	layerActivations = (ActivationFunction.Sigmoid, 
						ActivationFunction.Sigmoid)
	

	MLP = MultiLayerPerceptron(Dimensions, layerActivations, learningRate = 0.1, momentum = 0.8)
	#MLP = MultiLayerPerceptron(Dimensions, layerActivations)

	MLP.TrainEpoch(XTrain, YTrain, 1)

	# ErrorEpochList = []
	# #MLP.BatchTrain(XTrain,YTrain,120)
	# error = 100
	# #while error > 1e-5:
	# for j in range(1000):

	# 	rp = np.random.permutation(iris.data.shape[0])
	# 	X = Data[rp,:]
	# 	Y = iris.target[rp]
	# 	CleanedLabels = OneHotEncode(Y)
	# 	XTrain = X[:120]
	# 	YTrain = Y[:120]

	# 	errorlist = []
		
	# 	for i in range(120):
	# 		error = MLP.BackwardPass(XTrain[i],YTrain[i])
	# 		errorlist.append(error)
	# 	avgError = np.average(errorlist)
	# 	ErrorEpochList.append(avgError)
	# 	if (j % 100) == 0:
	# 		print("The average error is {}".format(avgError))
	# avgErrorEpoch = np.average(ErrorEpochList)
	# print("The average error over all Epochs is {}".format(avgErrorEpoch))



	# # for j in range(10):
	# # 	for i in range(4):
	# # 		MLP.BatchTrain(XTrain[i*30:(i+1)*30],YTrain[i*30:(i+1)*30],30)

	# rp = np.random.permutation(iris.data.shape[0])
	# X = Data[rp,:]
	# Y = iris.target[rp]
	# XTest = X[120:]
	# CleanedLabels = OneHotEncode(Y)
	# YTest = CleanedLabels[120:]

	# pred = MLP.TestData(XTest)

	# correct = NumberCorrect(pred, Y[120:])

	# print("The MLP got {} correct".format(correct))




	# # Dimensions = (5,4,3,3)
	# # layerActivations = (ActivationFunction.Sigmoid, 
	# # 					ActivationFunction.Sigmoid, 
	# # 					ActivationFunction.Sigmoid)

	# # MLP = MultiLayerPerceptron(Dimensions, layerActivations)
	# # Input = np.random.uniform(4,5,size=(5,1))
	# # MLP.ForwardPass(Input)
	# # #target = np.array([0.95,0]).reshape(2,1)
	# # target = np.array([0])
	# # cleanedTarget = MLP.OneHotEncode(target)


	# # # temp = np.array([0.95,0]).reshape(2,1)
	# # # for i in range(10):
	# # # 	target.append(temp)
	
	# # # for i in range(2):
	# # # 	print(MLP.BackwardPass(Input, target))

	# # weightdelta = MLP.BackwardPass(Input, cleanedTarget, batch=True)
		
	# # test = MLP.BatchTrain(Input, cleanedTarget, 1)