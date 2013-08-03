import math
import random
from itertools import permutations, izip_longest
from collections import Counter, defaultdict
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy import Inf

def are_mixable(v1,v2):
    """
    Input:
	v1,v2: Two dataseries
    Output: 
	    bool, True if the discrete labels are compatible between the Series.
	"""
    if set(v1).issubset(v2) or set(v2).issubset(v1): return True
    else: return False

def is_discrete(x):
    """
    Determine if a data series appears to be discrete, based on proportion of unique values.

    If the number of unique values is <= 5% of the number of observations, this will return True.
    Returns: 
	bool
    """
    try:
	if x.dtype == np.dtype(object): return True
    except:
	pass
    try:
	[ float(i) for i in x ]
    except ValueError:
	return True
    nclasses = len(Counter(x))
    if nclasses-1 <= (len(x)/20)+1: return True

def bc_continuous(d1, d2):
    """
    Input:
	d1, d1: Two continuous data series (Series).
    Returns:
	bc metric (float)
    """
    a = gaussian_kde(d1)
    b = gaussian_kde(d2)
    bc, err = quad( lambda x: math.sqrt(a(x) * b(x)), -Inf, Inf)
    if bc != 0.0:
	return -1*math.log(bc)
    else:
	return None

def bc_discrete(count1, count2):
    keys = set(count1.keys()).union( set(count2.keys()) )
    metric = 0
    for key in keys:
	if key not in count1: 
	    p = 0
	else: 
	    p = count1[key]/float(len(a))
	if key not in count2:
	    q = 0
	else: 
	    q = count2[key]/float(len(b))
	metric += math.sqrt( p*q )
    if metric != 0.0:
	return -1*math.log(metric)
    else: return None

def bc_discrete_map(count1, count2, map):
    """
    Input: 
	count1, count2: Frequencies for two discrete data distributions (Counter).
	map: A mapping of categorical variables from count1 to count2 (dict)

    Returns:
	bc statistic (float)
    """
    metric = 0
    for key in count1.keys():
	key2 = map[key]
	if key not in count1: 
	    p = 0
	else: 
	    p = count1[key]/float(sum(count1.values()))
	if key2 not in count2:
	    q = 0
	else: 
	    q = count2[key2]/float(sum(count2.values()))
	metric += math.sqrt( p*q )
    if metric != 0.0:
	return -1*math.log(metric)
    else: return None

def bc_discrete_best(v1,v2):
    """
    Input: v1,v2: 
	Two discrete data series (Series or any Iterable).

    Returns: 
	min_m (float): 
	    The minimum bc metric (float) between the two discrete distributions,
	    trying all mapping of catgorical values between the two series, 

	min_p (dict): 
	    The mapping that produced this statistic.
    """
    count1 = Counter(v1)
    count2 = Counter(v2)
    if len(count1) > len(count2):
	largest = count1
	smallest = count2
    else:
	largest = count2
	smallest = count1
    min_m = None
    for p in permutations(largest.keys()):
	pairing = dict(izip_longest(p, smallest.keys(), fillvalue = None ))
	m = bc_discrete_map(largest, smallest, pairing)
	if m < min_m or min_m == None: min_m = m; min_p = pairing
    return min_m, min_p

def best_mapping(a,b):
    """
    Input: 
	a,b: Two pandas data frame objects, or any dict of Iterable objects.

    Output: 
	A mapping of column names from dataframe b to names from dataframe a (dict).

	A mapping of discrete category names for data series in b to the category names
	used in the corresponding data series in a (dict).
    """
    d = defaultdict(dict)
    c = defaultdict(dict)
    for (k1,v1) in a.iteritems():
	for (k2,v2) in b.iteritems():
	    if not is_discrete(v1) and not is_discrete(v2):
		c[k1][k2] = bc_continuous(v1,v2)
	    elif is_discrete(v1) and is_discrete(v2):
		d[k1][k2] = bc_discrete_best(v1,v2)
    output = {}
    mappers = {}
    for key in c:
	name = min(c[key].items(), key = lambda x: x[1])[0]
	output[name] = key
    for key in d:
	top = min(d[key].items(), key = lambda x: x[1][0])
	name = top[0]
	mapper = top[1][1]
	output[name] = key
	mappers[name] = mapper
    return output, mappers


