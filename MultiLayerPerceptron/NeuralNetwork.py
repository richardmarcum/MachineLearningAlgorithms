#!/usr/bin/python3

import numpy as np
import math
from scipy.special import expit



def NumberCorrect(Prediction, TrueValues):

	numCorrect = 0
	for i in range(len(Prediction)):
		if Prediction[i] == TrueValues[i]:
			numCorrect = numCorrect + 1
	return numCorrect

class ActivationFunction:

	def Sigmoid(x, Derivative=False):
		output = expit(x)
		if not Derivative:
			return output
		else:
			return output*(1-output)

	def Linear(x, Derivative=False):
		if not Derivative:
			return x
		else:
			return 1.0

	def tanh(x, Derivative=False):
		if not Derivative:
			return np.tanh(x)
		else:
			return 1 - np.tanh(x)**2

	def Softmax(x, Derivative=False):
		output = np.exp(x) / np.sum(np.exp(x))
		if not Derivative:

			return output
		else:
			Size=x.shape[1]
			J = np.zeros((Size,Size))
			for i in range(Size):
				for j in range(Size):
					if i == j:
						J[i][j] = output[i] * (1 - output[j])
					else:
						J[i][j] = -output[i] * output[j]
			return J


class MultiLayerPerceptron:

	def __init__(self, Dimensions, layerActivations, learningRate=0.1,
					momentum=0.3, regularizer=0.25):

		self.layerFlist = layerActivations
		self.layerCount = len(Dimensions) - 1
		self.shape = Dimensions
		self.layerDim = []
		self.weights = []
		self.layerOutput = []
		self.layerInput = []
		self.learningRate = learningRate
		self.momentum = momentum
		self.oldWeightDelta = []
		self.oldAccumulatedWeightChange = []
		self.regularizer = regularizer

		self.local_grad = []

		for i in range(self.layerCount):
			if i == (self.layerCount-1):
				self.layerDim.append((self.shape[i+1],self.shape[i]+1))
			else:
				self.layerDim.append((self.shape[i+1]+1,self.shape[i]+1))

		for pairs in self.layerDim:
			lim = 1/math.sqrt(pairs[1])
			#lim = 0.05
			initWeights = np.random.uniform(low=-lim,high=lim,size=pairs)
			self.weights.append(initWeights)

		for weightList in self.weights:
			pair = weightList.shape
			zeros = np.zeros(pair)
			self.oldWeightDelta.append(zeros)

		for weightList in self.weights:
			pair = weightList.shape
			zeros = np.zeros(pair)
			self.oldAccumulatedWeightChange.append(zeros)


	def ForwardPass(self, Input):

		self.layerInput = []
		self.layerOutput = []

		Input = np.array(Input).reshape(self.shape[0],1)
		NumInputs = 1

		for location in range(self.layerCount):

			if location == 0:
				Input = np.vstack((np.ones((1,NumInputs)),Input))
				v = self.weights[location] @ Input
				#v[0]=1
				y = self.layerFlist[location](v)
				self.layerInput.append(v)
				self.layerOutput.append(y)
			else:
				v = (self.weights[location] @ self.layerOutput[-1] )
				#v[0]=1
				y = self.layerFlist[location](v)
				self.layerInput.append(v)
				self.layerOutput.append(y)

		return self.layerOutput[-1]


	def BackwardPass(self, Input, labels, batch=False):

		self.local_grad = []
		Input = np.array(Input).reshape((self.shape[0],1))
		NumInputs = Input.shape[1]
		self.ForwardPass(Input)
		weightDelta = []

		#This for loop computes all of the local gradients
		for index in reversed(range(self.layerCount)):

			if index == self.layerCount - 1:

				error_signal = labels - self.layerOutput[index]
				# print(self.layerOutput[index].shape)
				# print(labels.shape)
				# print(error_signal.shape)

				delta = (error_signal *
					self.layerFlist[index](self.layerInput[index], Derivative=True))

				self.local_grad.append(delta)

			else:
				#print(index)
				#print(self.layerCount-1-index)
				first = self.layerFlist[index](self.layerInput[index], Derivative=True)
				second = (self.weights[index+1].T
							@ self.local_grad[(self.layerCount - 1) - (index + 1)])
				delta = first * second
				self.local_grad.append(delta)

		#This for loop uses all of the local gradients to compute the weight
		#update matrices.
		for index in range(self.layerCount):

			if index == 0:

				Input = np.vstack((np.ones((1,NumInputs)),Input))
				weightDelta.append(self.local_grad[self.layerCount-1-index] @ Input.T)

			else:

				weightDelta.append(self.local_grad[self.layerCount-1-index]
												@ self.layerOutput[index-1].T)


		error = np.sum(error_signal ** 2)

		#this computes all of the weight delta matrices and if the batch flag is triggered
		#it will return the weightDelta matrix so they can be averaged and applied all at once
		# in the batch train method below.

		if batch == True:

			return weightDelta,error

		else:
			for i in range(len(self.weights)):
				self.weights[i] = self.weights[i] + self.learningRate * self.momentum * self.oldWeightDelta[i] + self.learningRate * weightDelta[i] - self.regularizer * self.learningRate * self.weights[i]
				self.oldWeightDelta = weightDelta[:]

			return error


	def BatchTrain(self, Input, labels, BatchSize):

		errorList = []
		AccumulatedWeightChange = []

		for weightList in self.weights:
			pair = weightList.shape
			zeros = np.zeros(pair)
			AccumulatedWeightChange.append(zeros)

		for i in range(BatchSize):
			currentWeightDelta,error = self.BackwardPass(Input[i], labels[i], batch=True)
			errorList.append(error)

			for j in range(self.layerCount):
				AccumulatedWeightChange[j] = AccumulatedWeightChange[j] + currentWeightDelta[j]

		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + self.learningRate * self.momentum * self.oldAccumulatedWeightChange[i] + self.learningRate * AccumulatedWeightChange[i] - self.regularizer * self.learningRate * self.weights[i]
			self.oldAccumulatedWeightChange = AccumulatedWeightChange[:]

		return np.average(errorList)

	def OneHotEncode(self, labels):

		numClasses = self.shape[-1]
		classList = (1,2,3)
		labelList = []


		for num in range(numClasses):
			temp = np.zeros((numClasses,1))
			temp[num] = 1
			labelList.append(temp)

		cleanedLabels = []

		for i in range(len(labels)):

			temp = labelList[int(labels[i][0]-1)]
			cleanedLabels.append(temp)

		return cleanedLabels

	def TrainEpoch(self, Data, Labels, numEpochs, BatchTrain=True):


		ErrorEpochList = []
		#errorlist = []

		for num in range(numEpochs):

			DataSize=Data.shape[0]
			rp = np.random.permutation(DataSize)
			X = Data[rp,:]
			Y = Labels[rp,:]
			if Y[0].shape[0] < 2:
				Y = self.OneHotEncode(Y)

			errorlist = []

			if BatchTrain:

				avgError = self.BatchTrain(X,Y, DataSize)
				print("The average error is {}".format(avgError))

			else:
				for i in range(DataSize):
					error = self.BackwardPass(X[i], Y[i])
					errorlist.append(error)
					print("The error is {}".format(error))
			#return errorlist
				avgError = np.average(errorlist)
				ErrorEpochList.append(avgError)

				if (num % 5) == 0:
					print("The average error is {}".format(avgError))

				avgErrorEpoch = np.average(ErrorEpochList)
				print("The average error over all Epochs is {}".format(avgErrorEpoch))

	def TestData(self, Input):

		prediction = []

		for i in range(Input.shape[0]):
			output = np.argmax(self.ForwardPass(Input[i]))+1
			prediction.append(output)

		return prediction

if __name__ == "__main__":

	print("You made it this far?")
