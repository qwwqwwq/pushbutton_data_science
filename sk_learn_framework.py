import sklearn
import sklearn.pipeline
import sklearn.linear_model
import sklearn.svm
import sklearn.metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math
import numpy as np
import sys
import random
import argparse
import pickle
import pandas
import sklearn_pandas
import logging
import numpy as np
from collections import Counter

### process command line args
parser = argparse.ArgumentParser(description='Use sklearn and pandas to perform supervised learning.')
parser.add_argument('--data', dest='trainfile', type=str,
                    help="Input training data.")
parser.add_argument('--save', dest='savefile', type=str,
                    help="Where model will be saved.")
parser.add_argument('--sample', type=float, dest='sample', default=0.75,
                    help="Proportion of data to train on.")
parser.add_argument('--predict', type=str, dest='predictfile',
                    help="Input testing data")
parser.add_argument('--log', type=str, dest='logfile', default='sklearn.log',
                    help="Log destination")
args = parser.parse_args()

##set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.logfile)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ih = logging.StreamHandler()
ih.setLevel(logging.INFO)
formatter_lite = logging.Formatter('%(asctime)s - %(message)s')

ch.setFormatter(formatter)
fh.setFormatter(formatter)
ih.setFormatter(formatter_lite)
logger.addHandler(ch)
logger.addHandler(fh)
logger.addHandler(ih)

class TrainTestDataset:
    """
    Class for a train/test dataset pair.
    """
    implicit_coercion_matrix = {np.dtype(np.int64) : np.dtype(np.float64) }
    coercion_matrix = {np.dtype(np.int64) : frozenset([np.dtype(np.float64)]) }
    preprocessing_matrix = { np.dtype(np.int64) : None, 
			     np.dtype(np.float64) : StandardScaler,
			     np.dtype(object) : LabelEncoder }
    acceptable_dtypes = { np.dtype(np.float64) : True }
    pipes = dict()
    cv_scores = dict()

    def __init__(self, train, test):
	self.load_training_data(train)
	self.load_testing_data(test)
	self.normalize_datasets()
	self.create_mapper()
    def coerce(self, series1, series2 ):
	"""
	If a coercsion from type 1 to type 2 is defined in self.coercion_matrix that allows two series to have the same dtype,
	force that conversion.

	Returns: NoneType
	"""
	assert isinstance(series1, pandas.core.series.Series) and isinstance(series2, pandas.core.series.Series)
	if self.can_coerce(series1.dtype, series2.dtype):
	    logger.info("Converting column (%s) in train from (%s) to (%s)." % (series1.name, series1.dtype, series2.dtype ) )
	    series1 = series1.astype(series2.dtype)
	elif self.can_coerce(series2.dtype, series1.dtype):
	    logger.info("Converting column (%s) in test from (%s) to (%s)." % (series2.name, series2.dtype, series1.dtype ) )
	    series2 = series2.astype(series1.dtype)
	return (series1, series2)
    def can_coerce(self, from_type, to_type):
	if to_type in self.coercion_matrix.get(from_type, set()):
	    return True
    def can_implicit_coerce(self, data_type):
	if data_type in self.implicit_coercion_matrix:
	    return True
    def implicit_coerce(self, series):
	return series.astype(self.implicit_coercion_matrix[series.dtype])
    def load_training_data(self, fn):
	logger.info("Begin loading training data from %s" % fn )
	self.train = pandas.io.parsers.read_table(fn, delim_whitespace=True, keep_default_na = True, na_values = ['NA'])
	logger.info("Finished loading training data from %s" % fn )
	data_type_summary = Counter( [ self.train[k].dtype for k in self.train.keys() ] )
	logger.info("Summary of training data types:\n\t%s" % data_type_summary )
    def load_testing_data(self, fn):
	logger.info("Begin loading testing data from %s" % fn )
	self.test = pandas.io.parsers.read_table(fn, delim_whitespace=True, keep_default_na = True, na_values = ['NA'])
	logger.info("Finished loading testing data from %s, rows: (%s), columns: (%s)" % (fn, self.test.shape[0], self.test.shape[1] ) )
	data_type_summary = Counter( [ self.test[k].dtype for k in self.test.keys() ] )
	logger.info("Summary of testing data types:\n\t%s" % data_type_summary )
    def normalize_datasets(self):
	columns_to_remove_train = set(self.train.keys()) - set(self.test.keys())
	columns_to_remove_test = set(self.test.keys()) - set(self.train.keys())
	common_columns =  set(self.test.keys()) & set(self.train.keys())
	for column in common_columns:
	    if self.train[column].dtype != self.test[column].dtype:
		self.train[column], self.test[column] = self.coerce(self.train[column], self.test[column] )
	    if self.train[column].dtype != self.test[column].dtype:
		logger.info("Column (%s) dtype (%s) in train != dtype (%s) in test, will remove." % (column,  self.test[column].dtype, self.train[column].dtype ) )
		columns_to_remove_train.add(column)
		columns_to_remove_test.add(column)
	    elif self.train[column].dtype == self.test[column].dtype and \
		    self.can_implicit_coerce(self.train[column].dtype):
		self.train[column] = self.implicit_coerce(self.train[column])
		self.test[column] = self.implicit_coerce(self.test[column])
	    if self.test[column].isnull().any():
		logger.info("Column (%s) dtype (%s) in test has NAs, will fill in with mean (%s)." % ( column, self.test[column].dtype, self.train[column].mean() ) )
		self.test[column] = self.test[column].fillna(self.test[column].mean())
	    if self.train[column].isnull().any():
		logger.info("Column (%s) dtype (%s) in train has NAs, will fill in with mean (%s)." % ( column, self.train[column].dtype, self.test[column].mean() ) )
		self.train[column] = self.train[column].fillna(self.train[column].mean())

	##for k in columns_to_remove_train: del self.train[k]
	logger.info("Removed %s columns from training data not present in test data." % len(columns_to_remove_train) )
	##for k in columns_to_remove_test: del self.test[k]
	logger.info("Removed %s columns from test data not present in training data." % len(columns_to_remove_test) )
	data_type_summary = Counter( [ self.test[k].dtype for k in self.test.keys() ] )
	logger.info("Summary of combined dataset types:\n\t%s" % data_type_summary )
	common_columns_names = (set(self.test.keys())-set(columns_to_remove_test)) & (set(self.train.keys())-set(columns_to_remove_train) )
	self.common_columns = dict( [(name, self.train[name].dtype) for name in common_columns_names ] )

    def create_mapper(self):
	preprocessing_mapping = [ (name, self.preprocessing_matrix[data_type]()) for name, data_type in self.common_columns.items() if data_type in self.acceptable_dtypes ]
	self.mapper = sklearn_pandas.DataFrameMapper(preprocessing_mapping)
	self.mapper.fit(self.train)

    def add_model(self, model):	
	self.pipes[str(model)] = sklearn.pipeline.Pipeline([
	('featurize', self.mapper),
	('m', model)])

    def cv(self):
	for name, pipe in self.pipes.iteritems():
	    logger.info("Conducting cross-validation with model (%s)." % name )
	    scores = sklearn_pandas.cross_val_score(pipe, self.train, self.train.votes, sklearn.metrics.mean_squared_error)
	    logger.info("Accuracy for (%s): %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() / 2))
	    self.cv_scores[name] = scores.mean()

    def predict_best(self):
	self.cv()
	name = sorted(self.cv_scores.items(), key = lambda x: x[1])[0][0]
	best_pipe = self.pipes[name]
	logger.info("Training top model (%s) on (%s) datapoints." % (name, self.train.shape[0]))
	best_pipe.fit(self.train, self.train.votes)
	logger.info("Using top model (%s) to predict (%s) datapoints." % (name, self.test.shape[0]))
	self.prediction = best_pipe.predict(self.test)
	print self.prediction

    def export_to_sklearn(self, columns = None):
	if not columns:
	    #TODO
	    pass


d = TrainTestDataset(args.trainfile, args.predictfile)
d.add_model(sklearn.linear_model.LinearRegression())
d.add_model(sklearn.svm.SVC(kernel='linear'))
d.add_model(sklearn.linear_model.BayesianRidge())
d.predict_best()


