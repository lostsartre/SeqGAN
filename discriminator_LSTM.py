import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(1234)

class Discriminator_LSTM(object):
	def __init__(self, sequence_length, vocab_size, embedding_size):
		self.model = Sequential()
		self.model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
		self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
		self.modeel.add(Dense(2, activation='sigmoid'))

