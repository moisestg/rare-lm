import tensorflow as tf
from multilayer_lstm import MultilayerLSTM
import data_utils
import analysis
import numpy as np

import time
import os
import sys
import collections


## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("train_path", "../simple-examples/data/ptb.train.txt", "Path to the training data") #/home/moises/thesis/lambada/lambada-dataset/train-novels/
tf.flags.DEFINE_string("test_path", "../simple-examples/data/ptb.test.txt", "Path to the test data") #/home/moises/thesis/lambada/lambada-dataset/train-novels/

# Model parameters
tf.flags.DEFINE_string("model_path", "", "Path to the trained model") 
#tf.flags.DEFINE_string("pretrained_emb", "word2vec", "Pretrained vectors to initialize the embedding matrix")
tf.flags.DEFINE_integer("emb_size", 200, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocab_size", 10000, "Size of the vocabulary")
tf.flags.DEFINE_integer("num_layers", 2, "Number of recurrent layers")
tf.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden & cell state")
tf.flags.DEFINE_integer("num_steps", 20, "Number of unrolled steps for BPTT")
tf.flags.DEFINE_string("optimizer", "grad_desc", "Optimizer used to calculate the gradients")
tf.flags.DEFINE_float("learning_rate", 1.0, "Learning rate of the optimizer")
tf.flags.DEFINE_float("learning_rate_decay", 0.5, "Decay (per epoch) of the learning rate")
tf.flags.DEFINE_float("keep_prob", 1.0, "Dropout output keep probability")
tf.flags.DEFINE_float("clip_norm", 5.0, "Norm value to clip the gradients")

# Eval parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.flags.DEFINE_boolean("plots", True, "Plot results splitted by categories")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in FLAGS.__flags.items():
	print("\t- {} = {}".format(attr.upper(), value))
print("\n")

## MAIN ##

# Load data
dataset = data_utils.LambadaDataset()
word2id, id2word = dataset.get_vocab(FLAGS.train_path, FLAGS.vocab_size)
test_data, max_len = dataset.get_test_data(FLAGS.test_path, word2id)


with tf.Graph().as_default():
	
	print("MAX LEN: "+str(max_len))
	FLAGS.num_steps = max_len
	test_input = dataset.get_test_batch_generator(config=FLAGS, data=test_data)
	with tf.variable_scope("model", reuse=None):
		model_test = MultilayerLSTM(is_training=False, config=FLAGS, pretrained_emb=None)

	saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

	sv = tf.train.Supervisor(logdir=None)
	with sv.managed_session() as session:

		saver.restore(session, FLAGS.model_path)
		session.run(tf.global_variables())

		"""
		print("\n\n** Trained model restored from: "+FLAGS.model_path+" **\n")
		test_losses, test_accs = dataset.eval_test(session, model_test, test_input)
		print("\n** Test Perplexity: %.3f Test accuracy: %.3f **\n" % (np.exp(np.mean(test_losses)), np.mean(test_accs)))
		"""
		test_pos = np.load("./analysis/test_pos.npy")
		dataset.eval_last_word_detailed(session, model_test, test_input, id2word, test_pos)
		# Plots
		if(FLAGS.plots):
			test_pos = np.load("./analysis/test_pos.npy")
			test_context = np.load("./analysis/test_context.npy")
			test_distance = np.load("./analysis/test_distance.npy")
			test_repetition = np.load("./analysis/test_repetition.npy")
			analysis.split_categories_plot(test_losses, test_accs, test_pos)
			analysis.split_categories_plot(test_losses, test_accs, test_context)
			analysis.split_categories_plot(test_losses, test_accs, test_distance)
			analysis.split_categories_plot(test_losses, test_accs, test_repetition)