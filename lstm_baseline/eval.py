import data_utils
from lm_model import BasicLSTM

import tensorflow as tf
import numpy as np
import os
import time
import datetime

# Data loading parameters
tf.flags.DEFINE_string("test_path", "/home/moises/thesis/lambada/lambada-dataset/test/", "Path to the test data")
# Model parameters
tf.flags.DEFINE_integer("vocab_size", 100000, "Size of the vocabulary (default: 100k)")
tf.flags.DEFINE_integer("num_steps", 20, "Number of steps for BPTT (default: 20)")
# Test parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1490130308/checkpoints/", "Checkpoint directory from training run")
# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in FLAGS.__flags.items():
	print("{}={}".format(attr.upper(), value))
print("")

## DATA PREPARATION ##

# Load data
print("Loading and preprocessing test dataset \n")
word2Id = data_utils.load_vocabulary("", FLAGS.vocab_size)
x_test, y_test = data_utils.load_data("test", FLAGS.num_steps, FLAGS.vocab_size, FLAGS.test_path, word2Id)
print("Done! \n")

## LOAD MODEL & EVALUATION LOOP ##

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# Load the saved meta graph and restore variables
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		# Get the placeholders from the graph by name
		input_x = graph.get_operation_by_name("input_x").outputs[0]
		input_y = graph.get_operation_by_name("input_y").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

		# Tensors we want to evaluate
		accuracies = graph.get_operation_by_name("stats/accuracy").outputs[0]
		losses = graph.get_operation_by_name("softmax/loss").outputs[0]

		# Generate batches for one epoch
		batches = data_utils.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)

		# Collect the stats here
		all_accuracies = []
		all_losses = []

		for batch in batches:
			x_batch, y_batch = zip(*batch)
			batch_accuracy, batch_loss = sess.run([accuracies, losses], {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0})
			all_accuracies = np.concatenate([all_accuracies, [batch_accuracy]])
			all_losses = np.concatenate([all_losses, [batch_loss]])

# Print evaluation stats
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(np.mean(all_accuracies)))
print("Perplexity: {:g}".format(np.exp(np.mean(all_perplexities)))) # CORPUS-WISE PERPLEXITY (1st AVERAGE, 2nd EXP)