import os
import sys

import tensorflow as tf

from generator import Generator
import pickle as pkl
import numpy as np

EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
VOCAB_SIZE = 7014 # 7014 from obama input.txt with min-frequency=5

def get_mapping(model_file, mapping_file):
    nearest = dict()

    generator = Generator(VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_file)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_file, ckpt_name))
        print("[*] Success to read {}".format(ckpt_name))

        embeddings = sess.run(generator.g_embeddings)

        for i in range(VOCAB_SIZE):
            dist = np.sum((embeddings - embeddings[i]) ** 2, axis=-1)
            near = np.argsort(dist)[1]
            nearest[i] = near

        pkl.dump(nearest, open(mapping_file,'wb+'))

    else:
        print("[*] Failed to find a checkpoint")

if __name__ == '__main__':
    get_mapping('./checkpoints/generator', 'save/nearest.pkl')