import math
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LinearLayer
from pybrain.datasets import SupervisedDataSet
import sys
import random
import argparse
import pickle
from sklearn.base import BaseEstimator
from pybrain.supervised.trainers import BackpropTrainer

class PybrainNN(BaseEstimator):
    def __init__(self, hidden_layers = (5,5), copy_X = True, max_epochs = 100):
	self.hidden_layers = hidden_layers
	self.max_epochs = max_epochs
	self.copy_X = copy_X
	self.trainer_class = BackpropTrainer

    def fit(X, y):
	self.shape = ( X.shape[1], y.shape[1] )
	self.network = buildNetwork( self.shape[0], *hidden_layers, self.shape[1])
	self.ds = SupervisedDataSet( *self.shape )
	for idx, r in enumerate(X):
	    ds.addSample(r, y[idx])
	self.trainer = self.trainer_class(self.network, self.ds)
	self.trainer.trainUntilConvergence( maxEpochs = self.max_epochs )

    def decision_function(X):
	assert X.shape[1] == self.shape[0]
	ds = SupervisedDataSet( *self.shape )
	for r in X:
	    ds.addSample(r, 0)
	return self.network.activateOnDataset(ds)

    def predict(X):
	return self.decision_function(X)

