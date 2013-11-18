import math
import heapq
import random
import re
from itertools import permutations, izip_longest
from collections import Counter, defaultdict
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy import Inf
import numpy as np
from numpy.linalg import LinAlgError

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
    nclasses = len(Counter(x))
    if nclasses-1 <= (len(x)/20)+1: return True

def bc_continuous(d1, d2):
    """
    Input:
	d1, d1: Two continuous data series (Series).
    Returns:
	bc metric (float)
    """
    d1=d1.dropna()
    d2=d2.dropna()
    if not len(d1) or not len(d2): return 0.0
    try:
	a = gaussian_kde(d1)
	b = gaussian_kde(d2)
	bc, err = quad( lambda x: math.sqrt(a(x) * b(x)), -Inf, Inf)
	return -1*math.log(bc)
    except LinAlgError, ZeroDivisonError:
	return Inf

def bc_discrete(count1, count2):
    keys = set(count1.keys()).union( set(count2.keys()) )
    a = sum(count1.values())
    b = sum(count2.values())
    metric = 0
    for key in keys:
	if key not in count1: 
	    p = 0
	else: 
	    p = count1[key]/float(a)
	if key not in count2:
	    q = 0
	else: 
	    q = count2[key]/float(b)
	metric += math.sqrt( p*q )
    if metric != 0.0:
	return -1*math.log(metric)
    else: return Inf

def bc_discrete_w(count1, w1, count2, w2):
    keys = set(count1.keys()).union( set(count2.keys()) )
    count1 = { k: (v/w1) for (k,v) in count1.iteritems() }
    count2 = { k: (v/w2) for (k,v) in count2.iteritems() }
    a = sum(count1.values())
    b = sum(count2.values())
    metric = 0
    for key in keys:
	if key not in count1: 
	    p = 0
	else: 
	    p = count1[key]/float(a)
	if key not in count2:
	    q = 0
	else: 
	    q = count2[key]/float(b)
	metric += math.sqrt( p*q )
    if metric != 0.0:
	return -1*math.log(metric)
    else: return Inf


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
    else: return Inf

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
	c1_first = True
    else:
	largest = count2
	smallest = count1
	c1_first = False
    pairing = dict(izip_longest([x[0] for x in sorted(largest.items(), key = lambda x: -x[1])], 
				[x[0] for x in sorted(smallest.items(), key = lambda x: -x[1])], 
				fillvalue = None ))
    m = bc_discrete_map(largest, smallest, pairing)
    inv_pairing = {v:k for k, v in pairing.items()}
    if c1_first:
	return m, inv_pairing
    else:
	return m, pairing

class Mapping(object):
    def __init__(self, k1, k2, bc):
	self.k1 = k1
	self.k2 = k2
	self.bc = bc
    def __lt__(self, other):
	return self.bc < other.bc

def best_mapping(a,b):
    """
    Input: 
	a,b: Two pandas data frame objects, or any dict of Iterable objects.

    Output: 
	A mapping of column names from dataframe b to names from dataframe a (dict).

	A mapping of discrete category names for data series in b to the category names
	used in the corresponding data series in a (dict).
    """
    d_heap = []
    c_heap = []
    for (k1,v1) in a.iteritems():
	for (k2,v2) in b.iteritems():
	    if not is_discrete(v1) and not is_discrete(v2):
		bc = bc_continuous(v1,v2)
		heapq.heappush(c_heap, (bc, k1, k2))
	    elif is_discrete(v1) and is_discrete(v2):
		bc, mapper = bc_discrete_best(v1,v2)
		heapq.heappush(d_heap, (bc, mapper, k1, k2))
    output = {}
    mappers = {}
    seen1 = set()
    seen2 = set()
    while c_heap:
	bc, k1, k2 = heapq.heappop(c_heap)
	if k1 in seen1 or k2 in seen2: 
	    continue
	else:
	    output[k2] = k1
	    seen1.add(k1)
	    seen2.add(k2)
    seen1 = set()
    seen2 = set()
    while d_heap:
	bc, mapper, k1, k2 = heapq.heappop(d_heap)
	if k1 in seen1 or k2 in seen2: 
	    continue
	else:
	    output[k2] = k1
	    mappers[k2] = mapper
	    seen1.add(k1)
	    seen2.add(k2)
    return output, mappers


