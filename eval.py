import tensorflow as tf
import argparse
from multilayer_lstm import MultilayerLSTM
import data_utils
import analysis
import numpy as np

import time
import os
import sys
import collections


## PARAMETERS ##
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Data loading parameters
parser.add_argument("--train_path", type=str, default="../simple-examples/data/ptb.train.txt", help="Path to the training data") #/home/moises/thesis/lambada/lambada-dataset/train-novels/
parser.add_argument("--test_path", type=str, default="../simple-examples/data/ptb.test.txt", help="Path to the test data") #/home/moises/thesis/lambada/lambada-dataset/dev/

# Model parameters
parser.add_argument("--model_path", type=str, default="", help="Path to the trained model") 
#tf.flags.DEFINE_string("pretrained_emb", "word2vec", "Pretrained vectors to initialize the embedding matrix")
parser.add_argument("--emb_size", type=int, default=200, help="Dimensionality of word embeddings")
parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary")
parser.add_argument("--num_layers", type=int, default=2, help="Number of recurrent layers")
parser.add_argument("--hidden_size", type=int, default=200, help="Size of the hidden & cell state")
parser.add_argument("--num_steps", type=int, default=20, help="Number of unrolled steps for BPTT")
parser.add_argument("--optimizer", type=str, default="grad_desc", help="Optimizer used to calculate the gradients")
parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate of the optimizer")
parser.add_argument("--learning_rate_decay", type=float, default=None, help="Decay (per epoch) of the learning rate") # 0.5
parser.add_argument("--keep_prob", type=float, default=1.0, help="Dropout output keep probability")
parser.add_argument("--clip_norm", type=float, default=5.0, help="Norm value to clip the gradients")

# Eval parameters
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--plots", type=bool, default=True, help="Plot results splitted by categories")

FLAGS, _ = parser.parse_known_args()

print("\n- Parameters:")
flags_list = sorted(vars(FLAGS))
for flag in flags_list:
	print("  --"+flag+"="+str(getattr(FLAGS, flag)))

## MAIN ##

# Load data
dataset = data_utils.LambadaDataset()
word2id, id2word = dataset.get_vocab(FLAGS.train_path, FLAGS.vocab_size)
test_data = dataset.get_test_data(FLAGS.test_path, word2id) #test_data, max_len = dataset.get_test_data(FLAGS.test_path, word2id)


with tf.Graph().as_default():
	
	#print("MAX LEN: "+str(max_len))
	#FLAGS.num_steps = max_len
	test_input = dataset.get_test_batch_generator(config=FLAGS, data=test_data)
	with tf.variable_scope("model", reuse=None):
		model_test = MultilayerLSTM(is_training=False, config=FLAGS, pretrained_emb=None)

	saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

	sv = tf.train.Supervisor(logdir=None)
	with sv.managed_session() as session:

		saver.restore(session, FLAGS.model_path)
		session.run(tf.global_variables())

		print("\n\n** Trained model restored from: "+FLAGS.model_path+" **\n")
		test_losses, test_accs = dataset.eval_test(session, model_test, test_input)
		print("\n** Test Perplexity: %.3f Test accuracy: %.3f **\n" % (np.exp(np.mean(test_losses)), np.mean(test_accs)))
		
		"""
		test_pos = np.load("./analysis/test_pos.npy")
		dataset.eval_detailed(session, model_test, test_input, id2word, test_pos)

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
		"""