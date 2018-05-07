import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Masking
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(1234)

class Discriminator_LSTM(object):
	def __init__(self, max_length, vocab_size, embedding_size, mask_val, dropout=0.2):
		self.model = Sequential()
		self.model.add(Masking(mask_value=mask_val, input_shape=(max_length,)))
		self.model.add(Embedding(vocab_size, embedding_size, input_length=max_length))
		self.model.add(LSTM(100, dropout=dropout, recurrent_dropout=dropout))
		self.model.add(Dense(2, activation='sigmoid'))

