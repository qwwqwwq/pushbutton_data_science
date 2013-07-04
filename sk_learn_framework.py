import sklearn, sklearn.pipeline, sklearn.linear_model, sklearn.svm, sklearn.metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math
import numpy as np
import sys
import random
import pickle
import pandas
import sklearn_pandas
import logging
import numpy as np
from collections import Counter

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
	##set up logging
	self.logger = logging.getLogger()
	self.logger.setLevel(logging.INFO)
	ih = logging.StreamHandler()
	ih.setLevel(logging.INFO)
	formatter_lite = logging.Formatter('%(asctime)s - %(message)s')
	ih.setFormatter(formatter_lite)
	self.logger.addHandler(ih)


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
	    self.logger.info("Converting column (%s) in train from (%s) to (%s)." % (series1.name, series1.dtype, series2.dtype ) )
	    series1 = series1.astype(series2.dtype)
	elif self.can_coerce(series2.dtype, series1.dtype):
	    self.logger.info("Converting column (%s) in test from (%s) to (%s)." % (series2.name, series2.dtype, series1.dtype ) )
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
	self.logger.info("Begin loading training data from %s" % fn )
	self.train = pandas.io.parsers.read_table(fn, delim_whitespace=True, keep_default_na = True, na_values = ['NA'])
	self.logger.info("Finished loading training data from %s" % fn )
	data_type_summary = Counter( [ self.train[k].dtype for k in self.train.keys() ] )
	self.logger.info("Summary of training data types:\n\t%s" % data_type_summary )
    def load_testing_data(self, fn):
	self.logger.info("Begin loading testing data from %s" % fn )
	self.test = pandas.io.parsers.read_table(fn, delim_whitespace=True, keep_default_na = True, na_values = ['NA'])
	self.logger.info("Finished loading testing data from %s, rows: (%s), columns: (%s)" % (fn, self.test.shape[0], self.test.shape[1] ) )
	data_type_summary = Counter( [ self.test[k].dtype for k in self.test.keys() ] )
	self.logger.info("Summary of testing data types:\n\t%s" % data_type_summary )
    def normalize_datasets(self):
	columns_to_remove_train = set(self.train.keys()) - set(self.test.keys())
	columns_to_remove_test = set(self.test.keys()) - set(self.train.keys())
	common_columns =  set(self.test.keys()) & set(self.train.keys())
	for column in common_columns:
	    if self.train[column].dtype != self.test[column].dtype:
		self.train[column], self.test[column] = self.coerce(self.train[column], self.test[column] )
	    if self.train[column].dtype != self.test[column].dtype:
		self.logger.info("Column (%s) dtype (%s) in train != dtype (%s) in test, will remove." % (column,  self.test[column].dtype, self.train[column].dtype ) )
		columns_to_remove_train.add(column)
		columns_to_remove_test.add(column)
	    elif self.train[column].dtype == self.test[column].dtype and \
		    self.can_implicit_coerce(self.train[column].dtype):
		self.train[column] = self.implicit_coerce(self.train[column])
		self.test[column] = self.implicit_coerce(self.test[column])
	    if self.test[column].isnull().any():
		self.logger.info("Column (%s) dtype (%s) in test has NAs, will fill in with mean (%s)." % ( column, self.test[column].dtype, self.train[column].mean() ) )
		self.test[column] = self.test[column].fillna(self.test[column].mean())
		if self.test[column].isnull().any(): 
		    self.logger.info("Column (%s) in test was unable to be imputed." % column )
		    columns_to_remove_test.add(column)
	    if self.train[column].isnull().any():
		self.logger.info("Column (%s) dtype (%s) in train has NAs, will fill in with mean (%s)." % ( column, self.train[column].dtype, self.test[column].mean() ) )
		self.train[column] = self.train[column].fillna(self.train[column].mean())
		if self.train[column].isnull().any():
		    self.logger.info("Column (%s) in train was unable to be imputed." % column )
		    columns_to_remove_train.add(column)
	    if self.train[column].dtype not in self.acceptable_dtypes: columns_to_remove_train.add(column)
	    if self.test[column].dtype not in self.acceptable_dtypes: columns_to_remove_test.add(column)
	self.logger.info("Removed %s columns from training data not present in test data." % len(columns_to_remove_train) )
	self.logger.info("Removed %s columns from test data not present in training data." % len(columns_to_remove_test) )
	common_columns_names = (set(self.test.keys())-set(columns_to_remove_test)) & (set(self.train.keys())-set(columns_to_remove_train) )
	assert not len(set(columns_to_remove_test) & common_columns_names )
	assert not len(set(columns_to_remove_train) & common_columns_names )
	self.common_columns = dict( [(name, self.train[name].dtype) for name in common_columns_names ] )
	self.logger.info("%s common features will be used." % len(self.common_columns) )
	data_type_summary = Counter( self.common_columns.values() )
	self.logger.info("Summary of combined dataset types:\n\t%s" % data_type_summary )

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
	    self.logger.info("Conducting cross-validation with model (%s)." % name )
	    scores = sklearn_pandas.cross_val_score(pipe, self.train, self.train.votes, sklearn.metrics.mean_squared_error)
	    self.logger.info("Accuracy for (%s): %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() / 2))
	    self.cv_scores[name] = scores.mean()

    def predict_best(self):
	self.cv()
	name = sorted(self.cv_scores.items(), key = lambda x: x[1])[0][0]
	best_pipe = self.pipes[name]
	self.logger.info("Training top model (%s) on (%s) datapoints." % (name, self.train.shape[0]))
	best_pipe.fit(self.train, self.train.votes)
	self.logger.info("Using top model (%s) to predict (%s) datapoints." % (name, self.test.shape[0]))
	self.prediction = best_pipe.predict(self.test)
