import os
import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator_LSTM import Discriminator_LSTM
from rollout import ROLLOUT
import pickle
from keras.models import load_model

import pdb

from nltk.translate.bleu_score import *
smooth_fn = SmoothingFunction().method2

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
VOCAB_SIZE = 7014 # 7014 from obama input.txt with min-frequency=5

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64
MASK_VAL = 0

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 20
positive_file = 'save/input.id.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_train_num = 64 * 5
generated_test_num =  64 * 2


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)+1):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w+') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None, auto_reweigh=False,
                emulate_multibleu=False):
    """
    Removed brevity penalty.

    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :param smoothing_function:
    :type smoothing_function: SmoothingFunction
    :param auto_reweigh:
    :type auto_reweigh: bool
    :param emulate_multibleu: bool
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    # bp = brevity_penalty(ref_lengths, hyp_lengths)
    bp = 1. # ignore brevity penalty

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = ( 1 / hyp_lengths ,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len, emulate_multibleu=emulate_multibleu)
    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    s =  bp * math.exp(math.fsum(s))
    return round(s, 4) if emulate_multibleu else s


def test_bleu(eval_file, ground_truth_file):
    def get_ground_truth(filename):
        fin = open(filename,'r')
        sents = []
        for line in fin:
            sent = line.strip().split()
            sents += sent
        return sents
    def get_eval(filename):
        fin = open(filename,'r')
        sents = []
        for line in fin:
            sent = line.strip().split()
            sents.append(sent)
        return sents
    gt = get_ground_truth(ground_truth_file)
    test_sents = get_eval(eval_file)
    global_bleu = corpus_bleu([[gt]] * len(test_sents), test_sents, smoothing_function=smooth_fn)
    return global_bleu


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        seq = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, seq)
        print('Iter {}, loss={}'.format(it, g_loss))
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    discriminator = Discriminator_LSTM(max_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE, embedding_size=dis_embedding_dim, mask_val=MASK_VAL)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')
    print ('Reading generator checkpoints...')
    ckpt = tf.train.get_checkpoint_state('./checkpoints/generator')
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join('./checkpoints/generator', ckpt_name))
        print("[*] Success to read {}".format(ckpt_name))
    else:
        print("[*] Failed to find a checkpoint")

        #  pre-train generator
        print('Start pre-training...')
        log.write('pre-training...\n')
        for epoch in range(PRE_EPOCH_NUM):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            print('Pretrain Epoch {}: loss = {}'.format(epoch, loss))
            if epoch == PRE_EPOCH_NUM-1:
                generate_samples(sess, generator, BATCH_SIZE, generated_test_num, eval_file)
                global_bleu = test_bleu(eval_file, positive_file)
                print('BLEU = {} (epoch {})'.format(global_bleu, epoch))
                log.write('BLEU = {} (epoch {})'.format(global_bleu, epoch))

        #save pre-trained generator
        modelName = 'generator.model'
        checkpointdir = './checkpoints/generator'
        if not os.path.exists(checkpointdir):
            os.makedirs(checkpointdir)
        saver.save(sess, os.path.join(checkpointdir, modelName), global_step=PRE_EPOCH_NUM)

    print('Start pre-training discriminator...')
    print ('reading discriminator...')

    if os.path.exists('./checkpoints/discriminator/discriminator.h5'):
        discriminator.model = load_model('./checkpoints/discriminator/discriminator.h5')
        print ('read successfully')
    else:
        print ('[*] Failed to load discriminator, begin to train...')
        for ep in range(10):
            print('Pretrain discriminator, epoch: {}'.format(ep))
            generate_samples(sess, generator, BATCH_SIZE, generated_train_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file, max_len=SEQ_LENGTH, mask_val=MASK_VAL)
            discriminator.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            for i in range(dis_data_loader.num_batch):
                X_train, Y_train = dis_data_loader.next_batch()
                discriminator.model.fit(X_train, Y_train, epochs=3, batch_size=BATCH_SIZE)

        if not os.path.exists('./checkpoints/discriminator'):
            os.makedirs('./checkpoints/discriminator')
        discriminator.model.save('./checkpoints/discriminator/discriminator.h5')

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        print('Batch {}'.format(total_batch))
        # Train the generator for one step
        print('Generator training ...')
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_test_num, eval_file)
            global_bleu = test_bleu(eval_file, positive_file)
            print('BLEU = {}'.format(global_bleu))
            log.write('BLEU = {}'.format(global_bleu))

        # Update roll-out parameters
        rollout.update_params()

        print('Discriminator training ...')
        for _ in range(3):
            generate_samples(sess, generator, BATCH_SIZE, generated_train_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file, nn_replace=True)
            discriminator.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            for i in range(dis_data_loader.num_batch):
                X_train, Y_train = dis_data_loader.next_batch()
                discriminator.model.fit(X_train, Y_train, epochs=3, batch_size=BATCH_SIZE)

    modelName = 'generator.model'
    checkpointdir = './checkpoints/generator'
    saver.save(sess, os.path.join(checkpointdir, modelName), global_step=PRE_EPOCH_NUM)

    discriminator.model.save('./checkpoints/discriminator/discriminator.h5')

    log.close()


if __name__ == '__main__':
    main()
