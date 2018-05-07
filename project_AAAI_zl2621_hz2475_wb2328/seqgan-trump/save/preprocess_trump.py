import sys
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import cPickle as pkl

def w2i_mapping(input_file, output_file, w2i_file):
	sents = map(word_tokenize, sent_tokenize(open(input_file,'r').read().decode('utf-8').lower()))
	print('Total: {} sentences.'.format(len(sents)))
	w2i = pkl.load(open('w2i.pkl','r'))
	ids = map(lambda l: map(lambda w: str(w2i[w]), l), sents)
	open(output_file,'w+').write('\n'.join(map(lambda l: ' '.join(l), ids)).encode('utf-8'))

def main():
	w2i_mapping(input_file=sys.argv[1], output_file=sys.argv[2], w2i_file=sys.argv[3])

if __name__ == '__main__':
	main()