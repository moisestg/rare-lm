import data_utils
from lm_model import BasicLSTM

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tqdm import tqdm

## PARAMETERS ##

# Data loading parameters
#tf.flags.DEFINE_float("dev_sample_percentage", .05, "Percentage of the training data used for validation (default: 5%)")
tf.flags.DEFINE_string("train_path", "/home/moises/thesis/lambada/lambada-dataset/train-novels/", "Path to the training data")
tf.flags.DEFINE_string("dev_path", "/home/moises/thesis/lambada/lambada-dataset/dev/", "Path to the dev data")
# Model parameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embeddings (default: 128)")
tf.flags.DEFINE_integer("vocab_size", 100000, "Size of the vocabulary (default: 100k)")
tf.flags.DEFINE_integer("num_layers", 1, "Number of recurrent layers (default: 1)")
tf.flags.DEFINE_integer("state_size", 512, "Size of the hidden & cell state (default: 512)")
tf.flags.DEFINE_integer("num_steps", 20, "Number of steps for BPTT (default: 20)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of the optimizer (default: 1e-4)")
tf.flags.DEFINE_float("keep_prob", .5, "Dropout output keep probability (default: 50%)")
tf.flags.DEFINE_float("clip_norm", 5.0, "Norm value to clip the gradients (default: 5)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("restore_path", "", "Path to the model to resume the training (default: None)")
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

# Load train & dev data
print("Loading and preprocessing training and dev datasets... \n")
word2Id = data_utils.load_vocabulary(FLAGS.train_path, FLAGS.vocab_size)
x_train, y_train = data_utils.load_data("train", FLAGS.num_steps, FLAGS.vocab_size, FLAGS.train_path, word2Id)
x_dev, y_dev = data_utils.load_data("dev", FLAGS.num_steps, FLAGS.vocab_size, FLAGS.dev_path, word2Id)

# Generate training batches
batches = data_utils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, shuffle=False)
print("Done! \n")

## MODEL AND TRAINING PROCEDURE DEFINITION ##

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		# Initialize model
		model = BasicLSTM(
			vocab_size=FLAGS.vocab_size, 
			embedding_size=FLAGS.embedding_dim, 
			num_steps=FLAGS.num_steps, 
			state_size = FLAGS.state_size,
			num_layers = FLAGS.num_layers
		)

		# Define training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		grads_and_vars = optimizer.compute_gradients(model.loss)
		gradients, variables = zip(*grads_and_vars)
		gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip_norm)
		train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.summary.scalar("loss", model.loss)
		acc_summary = tf.summary.scalar("acc", model.accuracy)
		perp_summary = tf.summary.scalar("perplexity", model.perplexity)

		# Train Summaries
		train_summary_op = tf.summary.merge([loss_summary, acc_summary, perp_summary])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.summary.merge([loss_summary, acc_summary, perp_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

		# Initialize all variables (or restore)
		if FLAGS.restore_path != "":
			saver.restore(sess, FLAGS.restore_path)
			print("Model restored from: "+FLAGS.restore_path)
			sess.run(tf.global_variables())
		else:
			sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

		# Define training and dev steps (batch) 
		def train_step(x_batch, y_batch):
			"""
			A single training step
			"""
			global state_train # save state for next batch

			feed_dict = {
				model.input_x: x_batch,
				model.input_y: y_batch,
				model.dropout_keep_prob: FLAGS.keep_prob,
				model.init_state: state_train
			}

			_, step, summaries, loss, accuracy, perplexity, state_train = sess.run(
				[train_op, global_step, train_summary_op, model.loss, model.accuracy, model.perplexity, model.final_state], # BATCH-WISE PERPLEXITY
				feed_dict)

			time_str = datetime.datetime.now().isoformat()
			print("\n{}: step {}, loss {:g}, acc {:g}, perp {:g}".format(time_str, step, loss, accuracy, perplexity))
			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_dev, y_dev, current_step):
			"""
			Evaluates model on the whole dev set
			"""
			state_dev = np.zeros((FLAGS.num_layers, 2, FLAGS.batch_size, FLAGS.state_size))
			batch_loss, batch_acc, batch_perp = [], [], []
			batches = data_utils.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, shuffle=False) 
			for batch in batches:
				#print(state_dev)
				x_batch, y_batch = zip(*batch)
				feed_dict = {
					model.input_x: x_batch,
					model.input_y: y_batch,
					model.dropout_keep_prob: 1.0,
					model.init_state: state_dev
				}

				step, summaries, loss, accuracy, state_dev = sess.run(
					[global_step, dev_summary_op, model.loss, model.accuracy, model.final_state],
					feed_dict)
				batch_loss.append(loss)
				batch_acc.append(accuracy)
				#batch_perp.append(perplexity)
			loss_value = tf.Summary.Value(tag="loss", simple_value=float(np.mean(batch_loss)))
			acc_value = tf.Summary.Value(tag="acc", simple_value=float(np.mean(batch_acc)))
			perp_value = tf.Summary.Value(tag="perplexity", simple_value=float(np.exp(np.mean(batch_loss)))) # CORPUS-WISE PERPLEXITY (1st AVERAGE, 2nd EXP)
			new_summ = tf.Summary()
			new_summ.value.extend([loss_value, acc_value, perp_value])
			dev_summary_writer.add_summary(new_summ, current_step)
			print("Accuracy: "+str(np.mean(batch_acc))+" Loss: "+str(np.mean(batch_loss))+" Perplexity: "+str(np.exp(np.mean(batch_loss))))
			print("")

		## TRAINING LOOP ##
		state_train = np.zeros((FLAGS.num_layers, 2, FLAGS.batch_size, FLAGS.state_size)) # initial state
		for batch in tqdm(batches, total = (int((len(x_train)-1)/FLAGS.batch_size) + 1)*FLAGS.num_epochs, initial=1):
			x_batch, y_batch = zip(*batch)
			train_step(x_batch, y_batch)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.evaluate_every == 0:
				print("\nEvaluation:")
				dev_step(x_dev, y_dev, current_step)
				print("")
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))