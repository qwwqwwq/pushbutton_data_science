import scipy.stats
from collections import Counter, defaultdict
import string
import sys

class DetermineHeader(object):
    """
    Determine if the first line of a tabular like file is a header
    based on the character frequencies in each line of the file.
    """
    def __init__(self, file):
	self.other_line_freqs = defaultdict(lambda: 0)
	self.first_line_freqs = defaultdict(lambda: 0)
	self.num_lines = 0
	## allow file_paths and open files as input
	if hasattr(file, 'read'):
	    self.process_file(file.read())
	else:
	    with open(file) as f:
		self.process_file(f.read())

    def process_file(self, data):
	f = ( line for line in data.splitlines() )
	first_line = f.next()
	self.update_freqs(self.first_line_freqs, first_line)
	for line in f:
	    self.update_freqs(self.other_line_freqs, line)
	    self.num_lines += 1

    def update_freqs(self, dest, line):
	for char, appearances in Counter(line).iteritems():
	    dest[char] += appearances

    def is_first_line_a_header(self):
	zero_in_both = [ x for x in string.printable if not (self.first_line_freqs[x] or self.other_line_freqs[x]) ] 
	first_line_vec = [ self.first_line_freqs[x] for x in string.printable if x not in zero_in_both ]
	other_line_vec = [ self.other_line_freqs[x]/float(self.num_lines) for x in string.printable if x not in zero_in_both ]
	chisq, P = scipy.stats.chisquare( first_line_vec, other_line_vec )
	## if P meets significance, first line has significantly
	## different character frequencies and is likely a header
	if P < 0.05: return 1
	else: return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "Determines if the the first line of a "
						   "tabular file appears to be a header row.")
    parser.add_argument("file", help="Input file path.")
    args = parser.parse_args()
    dh = DetermineHeader(args.file)
    print dh.is_first_line_a_header()
