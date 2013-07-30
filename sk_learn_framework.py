import sklearn, sklearn.pipeline, sklearn.linear_model, sklearn.svm, sklearn.metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer, LabelBinarizer
from sklearn.feature_selection import SelectKBest, RFECV
import math
import numpy as np
import sys
import random
import pandas
import pickle
from sklearn_pandas_JQ import *
import logging
import numpy as np
import scipy.stats
from collections import Counter
from Rank_Scaler import RankScaler
from flexible_feature_selector import flexible_scoring_function
import mrjob
from mrjob.job import MRJob

def H_ratio(x):
    """
    This metric is the ratio of the entropy of oberved sequence over the maximal entropy of a
    sequence of the same length. Can be thought of as the divergence of the observed sequence
    from the uniform distribution. The reason for this metric is to have a consistant way to
    define a sequence which is close to uniform. Variables that are close to uniform over the training
    data should be pruned from the model.
    """
    H = 0
    c = Counter(x)
    total = len(x)
    for item, counts in c.iteritems():
	H += -1*( math.log( counts / float(total) ) * ( counts / float(total) ))
    return H/math.log(total)

def is_normal(x):
    """
    Use shapiro test with alpha = 0.05 to test normality of sample
    """
    shaprio_results = scipy.stats.shapiro(x)
    if x[1] <= 0.05: return True

def is_discrete(x):
    """
    Determine if a data series appears to be discrete, based on proportion of unique values.

    If the number of unique values is <= 5% of the number of observations, this will return True.
    """
    if x.dtype == np.dtype(object): return True
    nclasses = len(Counter(x))
    if nclasses <= (len(x)/20): return True

class TrainTestDataset:
    """
    Class for a train/test dataset pair.
    """
    implicit_coercion_matrix = {np.dtype(np.int64) : np.dtype(np.float64) }
    coercion_matrix = {np.dtype(np.int64) : frozenset([np.dtype(np.float64)]) }
    preprocessing_matrix = { np.dtype(np.int64) : None, 
			     np.dtype(np.float64) : StandardScaler,
			     np.dtype(object) : LabelEncoder}
    acceptable_dtypes = { np.dtype(np.float64) : True, np.dtype(object) : True }
    pipes = dict()
    cv_scores = dict()

    def __init__(self, train, test=None, Kfeats =5):
	##set up logging
	self.logger = logging.getLogger()
	self.logger.handlers = []
	self.logger.setLevel(logging.INFO)
	ih = logging.StreamHandler()
	ih.setLevel(logging.INFO)
	formatter_lite = logging.Formatter('%(asctime)s - %(message)s')
	ih.setFormatter(formatter_lite)
	self.logger.addHandler(ih)
	self.H_ratio_max = 0.50
	self.common_columns = None
	self.Kfeats = Kfeats
	self.excluded_features = set()

	self.load_training_data(train)
	if test:
	    self.load_testing_data(test)
	    self.normalize_datasets()
	else:
	    self.test = None
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
	self.train = self.clean_data(pandas.io.parsers.read_table(fn, delim_whitespace=True, keep_default_na = True, na_values = ['NA']))
	self.logger.info("Finished loading training data from %s" % fn )
	data_type_summary = Counter( [ self.train[k].dtype for k in self.train.keys() ] )
	self.logger.info("Summary of training data types:\n\t%s" % data_type_summary )
	target_name = None
	while target_name not in self.train.keys():
	    target_name = raw_input("Which column is the variable to predict? ")
	self.training_target = self.train[target_name]
	self.logger.info("")

    def load_testing_data(self, fn):
	self.logger.info("Begin loading testing data from %s" % fn )
	self.test = self.clean_data(pandas.io.parsers.read_table(fn, delim_whitespace=True, keep_default_na = True, na_values = ['NA']))
	self.logger.info("Finished loading testing data from %s, rows: (%s), columns: (%s)" % (fn, self.test.shape[0], self.test.shape[1] ) )
	data_type_summary = Counter( [ self.test[k].dtype for k in self.test.keys() ] )
	self.logger.info("Summary of testing data types:\n\t%s" % data_type_summary )

    def clean_data(self, df):
	"""
	Impute NA values, remove unwanted data types.
	"""
	for feature in df.keys():
	    if self.can_implicit_coerce(df[feature].dtype):
		df[feature] = self.implicit_coerce(df[feature])
	    if df[feature].dtype not in self.acceptable_dtypes: 
		self.excluded_features.add(feature)
	    if df[feature].isnull().any():
		self.logger.info("Feature (%s) dtype (%s) in test has NAs, will fill in with mean (%s)." % ( feature, df[feature].dtype, df[feature].mean() ) )
		df[feature] = df[feature].fillna(df[feature].mean())
		if df[feature].isnull().any(): 
		    self.logger.info("Feature (%s) in test was unable to be imputed." % feature )
		    self.excluded_features.add(feature)
	return df

    def normalize_datasets(self):
	common_columns = set(self.test.keys()) & set(self.train.keys())
	for column in common_columns:
	    if self.train[column].dtype != self.test[column].dtype:
		self.train[column], self.test[column] = self.coerce(self.train[column], self.test[column] )
	    if self.train[column].dtype != self.test[column].dtype:
		self.logger.info("Column (%s) dtype (%s) in train != dtype (%s) in test, will remove." % (column,  self.test[column].dtype, self.train[column].dtype ) )
		self.excluded_features.add(column)
	usable_features = (self.test.keys() & set(self.train.keys())-self.excluded_features)
	self.common_columns = dict( [(name, self.train[name].dtype) for name in usable_features ] )
	self.logger.info("%s common features will be used." % len(self.common_columns) )
	data_type_summary = Counter( self.common_columns.values() )
	self.logger.info("Summary of combined dataset types:\n\t%s" % data_type_summary )

    def create_mapper(self):
	self.preprocessing_mapping = []
	if not self.common_columns and not self.test:
	    self.common_columns = { name:self.train[name].dtype for name in self.train.keys() }
	for name, data_type in self.common_columns.items():
	    if data_type not in self.acceptable_dtypes: continue
	    series = self.train[name]
	    if len(Counter(series)) <= 1: continue ##non informative features excluded
	    if is_discrete(series):
		if len(Counter(series)) == len(series): continue ##non informative features excluded
		if H_ratio(series) > self.H_ratio_max: 
		    print name, "excluded to to H_ratio", H_ratio(series)
		    continue
		self.logger.info("Feature (%s) classified as discrete with (%s) levels, levels are:\n%s" % (name, series.nunique(), series.unique()) )
		self.preprocessing_mapping.append( (name, LabelBinarizer()) )
	    else:
		self.logger.info("Feature (%s) classified as continuous with mean (%s) and variance (%s)." % (name, series.mean(), series.var()) )
		if is_normal(series):
		    self.logger.info("Feature (%s) normal by shapiro test." % (name) )
		    self.preprocessing_mapping.append( (name, StandardScaler() ) )
		else:
		    self.logger.info("Feature (%s) cannot be proved normal by shapiro test." % (name) )
		    self.preprocessing_mapping.append( (name, RankScaler()) )
	self.mapper = DataFrameMapper(self.preprocessing_mapping)
	self.mapper.fit(self.train)
	self.feature_selector = SelectKBest(score_func = flexible_scoring_function, k = self.Kfeats)
	self.feature_selector.fit(self.mapper.transform(self.train), self.training_target )
	selected_feats = [ self.mapper.index_to_name[idx] for idx in self.feature_selector.get_support(indices = True ) ]
	self.logger.info("The following features have been selected: %s" % ( ', '.join(selected_feats) ) )

    def add_model(self, model):	
	self.pipes[str(model)] = sklearn.pipeline.Pipeline([
	('featurize', self.mapper),
	('select', self.feature_selector),
	('m', model)])

    def cv(self):
	for name, pipe in self.pipes.iteritems():
	    self.logger.info("Conducting cross-validation with model (%s)." % name )
	    scores = cross_val_score(pipe, self.train, self.training_target, sklearn.metrics.mean_squared_error)
	    self.logger.info("Accuracy for (%s): %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() / 2))
	    self.cv_scores[name] = scores.mean()

    def predict_best(self):
	self.cv()
	name = sorted(self.cv_scores.items(), key = lambda x: x[1])[0][0]
	best_pipe = self.pipes[name]
	self.logger.info("Training top model (%s) on (%s) datapoints." % (name, self.train.shape[0]))
	best_pipe.fit(self.train, self.training_target)
	self.logger.info("Using top model (%s) to predict (%s) datapoints." % (name, self.test.shape[0]))
	self.prediction = best_pipe.predict(self.test)

