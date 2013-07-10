from sklearn.feature_selection import *
import numpy as np
x = [[ 0,    0,    1,    0.01],
 [ 0,    0,    1,    0.15],
 [ 2,    0,    0,    0.29],
 [ 0,    1,    0,    0.43],
 [ 2,    0,    0,    0.58],
 [ 0,    0,    1,    0.72],
 [ 0,    0,    1,    0.86],
 [ 0,    1,    0,    1  ]]
x = np.array(x)

y = np.array([1,0,1,0,1,0,1,1])


def detect_type(a):
    uniq = np.unique(a)
    if len(uniq) == 2 and 1 in uniq and 0 in uniq:
	return 'disc'
    else:
	return 'cont'

def f_classif_rev(X, y):
    return f_classif(y, X)

def flexible_scoring_function(X, y):
    data_types = []
    for feature in X.T:
	data_types.append(detect_type(feature))
    response_type = detect_type(y)
    if response_type == 'cont':
	scoring_map = { 'cont' : f_regression, 'disc' : f_classif_rev }
    else:
	scoring_map = { 'cont' : f_classif, 'disc' : chi2 }
    output_scores = np.array([])
    output_P = np.array([])
    for idx, feature in enumerate(X.T):
	feature = np.reshape(feature, (len(feature), ) )
	stats =  scoring_map[data_types[idx]](feature, y) 
	output_scores = np.append(output_scores, stats[0])
	output_P = np.append(output_P, stats[1])
    print sorted([x for x in enumerate(1/output_P)], key = lambda x: x[1], reverse = True )[:10]
    return 1/output_P, output_P
  


#flexible_scoring_function(x, y)
