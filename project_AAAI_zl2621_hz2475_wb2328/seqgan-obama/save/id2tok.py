import cPickle as pkl
import sys

fin = open(sys.argv[1], 'r')
i2w = pkl.load(open('i2w.pkl','r'))
fout = open(sys.argv[1]+'.tok','w+')

for line in fin:
	ids = map(int, line.strip().split())
	toks = [i2w[i].encode('utf-8') for i in ids]
	fout.write(' '.join(toks) + '\n')
