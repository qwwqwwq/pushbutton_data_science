import pandas
import sys
import math
from bhattacharyya import best_mapping
import argparse

parser = argparse.ArgumentParser(description='Combine N spreadsheets into one intelligently.')
parser.add_argument('filenames', type=str, nargs='+',
    help="Input csv filenames.")
parser.add_argument('--use-labels', dest='use_labels',action='store_true', default=False,
    help="Don't examine two dataseries if they have the same name, assume they are the same.")
parser.add_argument('--outfile', dest='outfile', type=str, required = True,
    help="Output csv.")
args = parser.parse_args()

def combine_dfs(dfs, use_labels=False):
    dfs.sort(key = lambda x: len(x.index) )
    while len(dfs) > 1:
	df1 = dfs.pop(0)
	df2 = dfs.pop(0)
	name_map, data_map = best_mapping(df1, df2)
	for k,v in data_map.iteritems():
	    if use_labels and (k in df1): continue
	    df2[k] = df2[k].apply(lambda x: v[x])
	for k in name_map.keys():
	    if args.use_labels and (k in df1):
		name_map[k] = k
	df2.rename( columns=name_map, inplace=True)
	dfs.append(pandas.concat([df1, df2]))
    return dfs.pop()

if __name__ == '__main__':
    dfs=[]
    for fn in args.filenames:
	dfs.append( pandas.io.parsers.read_csv(fn) )
    combine_dfs(dfs, use_labels = args.use_labels ).to_csv(args.outfile, index=False)
