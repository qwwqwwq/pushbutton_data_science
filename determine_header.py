import scipy.stats
from collections import Counter, defaultdict
import string
import sys
from bhattacharyya import bc_discrete_w

class DetermineHeader(object):
    """
    Determine if the first line of a tabular like file is a header
    based on the character frequencies in each line of the file.
    """
    def __init__(self, file, sample_size = 100):
	self.sample_size = sample_size
	self.other_line_freqs = defaultdict(lambda: 0)
	self.first_line_freqs = defaultdict(lambda: 0)
	self.num_lines = 0
	self.bcs = []
	## allow file_paths and open files as input
	if hasattr(file, 'read'):
	    self.process_file(file.read())
	else:
	    with open(file) as f:
		self.process_file(f.read())

    def process_file(self, data):
	f = [ line for (idx,line) in enumerate(data.splitlines()) if idx < self.sample_size ]
	print f
	self.num_lines = len(f)
	for idx, line in enumerate(f):
	    first_line_freqs = defaultdict(lambda: 0)
	    other_line_freqs = defaultdict(lambda: 0)
	    line = f[idx]
	    self.update_freqs(first_line_freqs, line)
	    for idx2,line in enumerate(f):
		if idx2 != idx:
		    self.update_freqs(other_line_freqs, line)
	    if idx == 0:
		self.first_metric = bc_discrete_w(first_line_freqs, 1, other_line_freqs, (self.num_lines - 1))
		print self.first_metric
	    self.bcs.append(bc_discrete_w(first_line_freqs, 1, other_line_freqs, (self.num_lines - 1)))
	print self.bcs
	self.bcs.sort()
	print self.bcs

    def update_freqs(self, dest, line):
	for char, appearances in Counter(line).iteritems():
	    dest[char] += appearances

    def is_first_line_a_header(self):
	print self.bcs.index(self.first_metric)
	if self.bcs.index(self.first_metric) >= ((len(self.bcs)-1) * 0.95): return True
	else: return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "Determines if the the first line of a "
						   "tabular file appears to be a header row.")
    parser.add_argument("file", help="Input file path.")
    args = parser.parse_args()
    dh = DetermineHeader(args.file)
    print dh.is_first_line_a_header()
