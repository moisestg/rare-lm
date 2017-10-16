import tensorflow as tf
import numpy as np
import argparse
import os
import time

from mixture_lm import MixtureLM
import lambada_utils as utils

## PARAMETERS ##
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Data loading parameters
parser.add_argument("--train_path", type=str, help="Path to the training data")
parser.add_argument("--test_path", type=str, help="Path to the test data")

# Model parameters
parser.add_argument("--model_path", type=str, default="", help="Path to the trained model") 
#parser.add_argument("--pretrained_emb", type=str, default=None, help="Pretrained vectors to initialize the embedding matrix")
parser.add_argument("--emb_size", type=int, default=200, help="Dimensionality of word embeddings")
parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary")
parser.add_argument("--num_layers", type=int, default=1, help="Number of recurrent layers")
parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden & cell state")
parser.add_argument("--num_steps_eval", type=int, default=35, help="Number of unrolled steps (eval)")
parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate of the optimizer")
parser.add_argument("--keep_prob", type=float, default=1.0, help="Dropout output keep probability")
parser.add_argument("--clip_norm", type=float, default=5.0, help="Norm value to clip the gradients")
parser.add_argument("--lamda", type=float, default=0.0, help="Mixing parameter for the loss")

# Training parameters
parser.add_argument("--batch_size_eval", type=int, default=64, help="Batch size (eval)") 

config, _ = parser.parse_known_args()

print("\n- Parameters:")
config_list = sorted(vars(config))
for param in config_list:
	print("  --"+param+"="+str(getattr(config, param)))


## MAIN ##

# Load data
word2id, id2word = utils.get_vocab(config.train_path, config.vocab_size)
lm_testData = utils.SlidingGenerator(config.test_path, word2id, config.batch_size_eval, config.num_steps_eval)
switch_testData = utils.get_switchData(config.test_path, config.batch_size_eval, config.num_steps_eval)

with tf.Graph().as_default():

	with tf.variable_scope("model", reuse=None):
		model_test = MixtureLM(config=config, is_training=False, pretrained_emb=None)

	saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

	sv = tf.train.Supervisor(logdir=None)
	with sv.managed_session() as session:

		saver.restore(session, config.model_path)
		session.run(tf.global_variables())

		utils.eval_test(session, model_test, lm_testData, switch_testData, id2word)