import pandas
from collections import Counter
import sys
import math
from bhattacharyya import best_mapping
import argparse
import re
import numpy as np

parser = argparse.ArgumentParser(description='Combine N spreadsheets into one intelligently.')
parser.add_argument('filenames', type=str, nargs='+',
    help="Input csv filenames.")
parser.add_argument('--use-labels', dest='use_labels',action='store_true', default=False,
    help="Don't examine two dataseries if they have the same name, assume they are the same.")
parser.add_argument('--coerce-numeric', dest='coerce_numeric',action='store_true', default=False,
    help="If a series contains strings which appear to be numeric, coerce to a numeric series.")
parser.add_argument('--unify-categorical', dest='unify_categorical',action='store_true', default=True,
    help="Modify cateorical variable names based on relative frequency so that the same set of categorical variable names is used in all datasets.")
parser.add_argument('--outfile', dest='outfile', type=str, required = True,
    help="Output csv.")
args = parser.parse_args()

def coerce_numeric(v):
    if isinstance(v, basestring):
	stripv = v.strip()
	try:
	    float(v)
	except ValueError:
	    return None
	return stripv
    else:
	if np.isnan(v): return None
	raise TypeError("Non-string object detected in df {t}.".format(t = type(v)))

def coerce_numeric_series(df):
    for key in df.keys():
	nclasses = len(Counter(df[key]))
	if nclasses-1 <= (len(df[key])/20)+1: continue
	if df[key].dtype == np.dtype('O'):
	    coerced = df[key].apply(coerce_numeric)
	    coerced = coerced.astype(float)
	    if coerced.count() < (len(coerced) * float(0.50)):
		continue
	    else:
		df[key] = coerced

def enforce_existing_label_mapping(mapper, S2):
    remove = []
    S2 = set(S2)
    for k,v in mapper.iteritems():
	try:
	    if k in S2 and k != v:
		remove.append(k)
	except Exception as e:
	    if k is np.nan:
		remove.append(k)
    for k in remove:
	mapper[k] = k
    return mapper

def combine_dfs(dfs, use_labels=False, unify_categorical=False):
    dfs.sort(key = lambda x: len(x.index) )
    while len(dfs) > 1:
	df1 = dfs.pop(0)
	df2 = dfs.pop(0)
	name_map, data_map = best_mapping(df1, df2)
	for k,v in data_map.iteritems():
	    if use_labels and (k in df1): continue
	    if len(Counter(df1[name_map[k]])) == len(Counter(df2[k])) and unify_categorical:
		v = enforce_existing_label_mapping(v, df2[k])
		df2[k] = df2[k].apply(lambda x: v.setdefault(x, x) )
	for k in name_map.keys():
	    if args.use_labels and (k in df1):
		del name_map[k]
	df2 = df2[name_map.keys()]
	df2.rename( columns=name_map, inplace=True)
	print "Attribute mapping:"
	print name_map

	df2.describe()
	df1.describe()
	dfs.append(pandas.concat([df1, df2], keys = args.filenames))
    return dfs.pop()

if __name__ == '__main__':
    dfs=[]
    for fn in args.filenames:
	dfs.append( pandas.io.parsers.read_csv(fn, sep=None) )
	print "Description of {fn}".format(fn=fn)
	print dfs[-1].describe()
    if args.coerce_numeric:
	for df in dfs:
	    coerce_numeric_series(df)
    combine_dfs(dfs, use_labels = args.use_labels, unify_categorical = args.unify_categorical ).to_csv(args.outfile, index=False)
