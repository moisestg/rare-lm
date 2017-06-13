import tensorflow as tf
from multilayer_lstm import MultilayerLSTM
import data_utils
import numpy as np
import time
import os
import sys
import collections


## PARAMETERS ##

# Data loading parameters
tf.flags.DEFINE_string("train_path", "../simple-examples/data/ptb.train.txt", "Path to the training data") #/home/moises/thesis/lambada/lambada-dataset/train-novels/
tf.flags.DEFINE_string("dev_path", "../simple-examples/data/ptb.valid.txt", "Path to the dev data") #/home/moises/thesis/lambada/lambada-dataset/dev/
tf.flags.DEFINE_string("save_path", "./runs/", "Path to save the model's checkpoints and summaries")

# Model parameters
tf.flags.DEFINE_string("pretrained_emb", None, "Pretrained vectors to initialize the embedding matrix")
tf.flags.DEFINE_integer("emb_size", 200, "Dimensionality of word embeddings")
tf.flags.DEFINE_integer("vocab_size", 10000, "Size of the vocabulary")
tf.flags.DEFINE_integer("num_layers", 2, "Number of recurrent layers")
tf.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden & cell state")
tf.flags.DEFINE_integer("num_steps", 20, "Number of unrolled steps for BPTT")
tf.flags.DEFINE_string("optimizer", "grad_desc", "Optimizer used to calculate the gradients")
tf.flags.DEFINE_float("learning_rate", 1.0, "Learning rate of the optimizer")
tf.flags.DEFINE_float("learning_rate_decay", None, "Decay (per epoch) of the learning rate") # 0.5
tf.flags.DEFINE_float("keep_prob", 1.0, "Dropout output keep probability")
tf.flags.DEFINE_float("clip_norm", 5.0, "Norm value to clip the gradients")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch size") 
tf.flags.DEFINE_integer("num_epochs", 13, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps ")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("restore_path", None, "Path to the model to resume the training (default: None)")

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
train_data = dataset.get_train_data(FLAGS.train_path, word2id)
valid_data, max_len = dataset.get_dev_data(FLAGS.dev_path, word2id)


# Load pretrained embeddings (if any)
if FLAGS.pretrained_emb == "word2vec":
	pretrained_emb = data_utils.get_word2vec(FLAGS.train_path, FLAGS.emb_size, word2id)
else:
	pretrained_emb = None


# Define graph
with tf.Graph().as_default():
	initializer = tf.random_uniform_initializer(-.05, .05) # Default variables initializer

	# Initialize the model 
	train_input = dataset.get_train_batch_generator(config=FLAGS, data=train_data)
	with tf.variable_scope("model", reuse=None, initializer=initializer):
		model_train = MultilayerLSTM(is_training=True, config=FLAGS, pretrained_emb=pretrained_emb)

	print("MAX LEN: "+str(max_len))
	FLAGS.num_steps = max_len
	valid_input = dataset.get_dev_batch_generator(config=FLAGS, data=valid_data)
	with tf.variable_scope("model", reuse=True, initializer=initializer):
		model_valid = MultilayerLSTM(is_training=False, config=FLAGS, pretrained_emb=pretrained_emb)

	# Define saver to checkpoint the model
	out_path = FLAGS.save_path + str(int(time.time()))
	checkpoint_path = os.path.abspath(os.path.join(out_path, "checkpoints"))
	checkpoint_prefix = os.path.join(checkpoint_path, "model")
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

	# Create session	
	sv = tf.train.Supervisor(logdir=None)
	with sv.managed_session() as session:

		## TRAIN LOOP ##

		# Restore previous model (if any) to resume training
		if FLAGS.restore_path is not None:
			saver.restore(session, FLAGS.restore_path)
			print("\n\n\n** Model restored from: "+FLAGS.restore_path+" **\n\n\n")
			session.run(tf.global_variables())

		# Define sumary writers
		train_summary_path = os.path.join(out_path, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_path, session.graph)
		dev_summary_path = os.path.join(out_path, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_path, session.graph)

		print("\n\n\n*** START TRAINING ***\n\n")

		for i in range(FLAGS.num_epochs):

			print("\nEpoch: %d" % (i + 1))

			# Decay of the learning rate (if any)
			if FLAGS.learning_rate_decay is not None:
				#lr_decay = FLAGS.learning_rate_decay ** max(i + 1 - 4, 0.0)
				lr_decay = FLAGS.learning_rate_decay
				model_train.assign_lr(session, FLAGS.learning_rate * lr_decay)

			# Train epoch variables
			start_time = time.time()
			iters = 0
			state = session.run(model_train.initial_state, {model_train.batch_size: FLAGS.batch_size}) # initial state defined in the model

			fetches = {
				"cost": model_train.cost,
				"final_state": model_train.final_state,
				"accuracy": model_train.accuracy,
				"eval_op" : model_train.train_op,
			}

			# Iterate through all batches (one epoch)
			for step in range(train_input.epoch_size): 
				
				input_x, input_y = train_input.get_batch()
				feed_dict = {
					model_train.input_x: input_x,
					model_train.input_y: input_y,
					model_train.batch_size: input_x.shape[0],
				}

				# Feed previous state
				for i, (c, h) in enumerate(model_train.initial_state):
					feed_dict[c] = state[i].c
					feed_dict[h] = state[i].h
				
				# Run batch
				results = session.run(fetches, feed_dict)
				cost = results["cost"]
				state = results["final_state"]
				accuracy = results["accuracy"]
				perplexity = np.exp(cost / train_input.num_steps)
				iters += train_input.num_steps

				current_step = sv.global_step.eval(session) 

				# Print some info
				if step % 100 == 0: # epoch_size = 90687 / (model_train.input.epoch_size // 10) == 10
					print("Step: %i: Perplexity: %.3f, Accuracy: %.3f, Speed: %.0f wps" %
								(current_step, perplexity, accuracy,
								 iters * train_input.batch_size / (time.time() - start_time)))

				# Write train summary 
				if step % 1000 == 0: # (model_train.input.epoch_size // 10) == 10
					data_utils.write_summary(train_summary_writer, current_step, {"perplexity": perplexity, "accuracy": accuracy})

				# Eval on dev set
				if current_step % FLAGS.evaluate_every == 0:
					valid_losses, valid_accs = dataset.eval_dev(session, model_valid, valid_input, dev_summary_writer)
					print("\n** Step: %i: Valid Perplexity: %.3f,  Valid Accuracy: %.3f**\n" % (current_step, np.exp(np.mean(valid_perp)), np.mean(valid_acc)))

				# Checkpoint model
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(session, checkpoint_prefix, global_step=current_step)
					print("\n** Saved model checkpoint to {} **\n".format(path))

			print("\n\n----- Last epoch took a total of: "+str(time.time() - start_time)+" s ------\n")