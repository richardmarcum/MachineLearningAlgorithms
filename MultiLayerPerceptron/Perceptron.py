#Script modeling Perceptron learning algorithm

import numpy as np
import matplotlib.pyplot as plt
import math 



def generateData(N, x, y, Sigma):
	'''generateData(N, x, y, Sigma): Generate N uniformly spaced data points with mean [1,1] and with variance Sigma'''
	data = np.random.multivariate_normal([x,y],Sigma*np.eye(2), N)
	return data

def plotData(data1,data2):
	'''Plot the data generated from the generateData function'''
	fig = plt.figure()
	plt.scatter(data1[:,0],data1[:,1], c='b', linewidth=0)
	plt.scatter(data2[:,0],data2[:,1], c='r', linewidth=0)
	plt.show()

def savePlot(data1,data2,Data, weights,n):
	#l = 0
	#dim = np.shape(Data)[0]*np.shape(Data)[1]
	#u = int(max(Data.reshape(dim,1)))+1
	xrange = np.arange(-1,4,0.001)
	line = -weights[0]/weights[1]*xrange + weights[2]/weights[1]
	fig = plt.figure()
	plt.scatter(data1[:,0],data1[:,1], c='b', linewidth=0)
	plt.scatter(data2[:,0],data2[:,1], c='r', linewidth=0)
	plt.plot(xrange,line, 'g')
	plt.show()

def labelData(data,label):
	N = np.shape(data)[0]
	data = np.c_[data, np.zeros(N)+label]
	return data

class Perceptron:
	"A basic Perceptron"

	def __init__(self,inputs,targets):
		"Construct the Perceptron"
		if np.ndim(inputs)>1:
			self.nIn = np.shape(inputs)[1]
		else:
			self.nIn = 1

		if np.ndim(targets)>1:
			self.nOut = np.shape(targets)[1]
		else:
			self.nOut = 1

		self.nData = np.shape(inputs)[0]

		#initialise the weights of the network
		self.weights = np.random.rand(self.nIn+1,self.nOut)

	def pcnfwd(self, inputs):
		"run the network forward"
		#compute activations
		activation = inputs @ self.weights
		#threshold the activations
		return np.where(activation>0,1,0)

	def pcntrain(self,inputs,targets,eta,nIterations):
		"train the perceptron"
		#add the bias node of the perceptron
		inputs = np.concatenate((inputs, -np.ones((self.nData,1))),axis=1)
		print("The input data is, ")
		print(inputs)
		#Training loops
		#change = range(self.nData)
		self.activations = np.ones((self.nData,1))+1
		for n in range(nIterations):
			if not np.array_equal(self.activations, targets):
				self.activations = self.pcnfwd(inputs);
				self.weights -= eta*inputs.T@(self.activations - targets)
				print("Iteration: ", n)
				print(self.weights)
				print("Final outputs are:")
				print(self.activations)
				savePlot(data1,data2,Data, self.weights,n)
		return self.weights



if __name__ == '__main__': 

	#set up testing data for perceptron
	Sigma = 1/50
	data1 = generateData(200,2.5,2.5,Sigma)
	data2 = generateData(200,1,1,Sigma)
	plotData(data1,data2)
	Data1 = labelData(data1,0)
	Data2 = labelData(data2,1)
	Data = np.concatenate((Data1,Data2),axis = 0)
	perceptron = Perceptron(Data[:,0:2],Data[:,2:])
	perceptron.pcntrain(Data[:,0:2],Data[:,2:],0.1,20)

