import tensorflow as tf
import argparse
from multilayer_lstm import MultilayerLSTM
import data_utils
import numpy as np
import time
import os
import sys
import collections


## PARAMETERS ##
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Data loading parameters
parser.add_argument("--train_path", type=str, default="../simple-examples/data/ptb.train.txt", help="Path to the training data") #/home/moises/thesis/lambada/lambada-dataset/train-novels/
parser.add_argument("--dev_path", type=str, default="../simple-examples/data/ptb.valid.txt", help="Path to the dev data") #/home/moises/thesis/lambada/lambada-dataset/dev/
parser.add_argument("--save_path", type=str, default="./runs/", help="Path to save the model's checkpoints and summaries")

# Model parameters
parser.add_argument("--pretrained_emb", type=str, default=None, help="Pretrained vectors to initialize the embedding matrix")
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

# Training parameters
parser.add_argument("--batch_size", type=int, default=128, help="Batch size") 
parser.add_argument("--num_epochs", type=int, default=13, help="Number of training epochs")
parser.add_argument("--evaluate_every", type=int, default=1000, help="Evaluate model on dev set after this many steps")
parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save model after this many steps ")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")
parser.add_argument("--restore_path", type=str, default=None, help="Path to the model to resume the training (default: None)")

FLAGS, _ = parser.parse_known_args()

print("\n- Parameters:")
flags_list = sorted(vars(FLAGS))
for flag in flags_list:
	print("  --"+flag+"="+str(getattr(FLAGS, flag)))


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

		prev_valid_perp = float("Inf") # for learning rate decay

		for i in range(FLAGS.num_epochs):

			print("\nEpoch: %d" % (i + 1))

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
					valid_perp, valid_acc = dataset.eval_dev(session, model_valid, valid_input, dev_summary_writer)
					#valid_perp = np.exp(np.mean(valid_losses))
					#valid_acc = np.mean(valid_accs)
					print("\n** Step: %i: Valid Perplexity: %.3f,  Valid Accuracy: %.3f**\n" % (current_step, valid_perp, valid_acc))
					# Decay of the learning rate (if any)
					if FLAGS.learning_rate_decay is not None and valid_perp > prev_valid_perp:
						model_train.assign_lr(session, FLAGS.learning_rate * FLAGS.learning_rate_decay)
					prev_valid_perp = valid_perp

				# Checkpoint model
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(session, checkpoint_prefix, global_step=current_step)
					print("\n** Saved model checkpoint to {} **\n".format(path))

			print("\n\n----- Last epoch took a total of: "+str(time.time() - start_time)+" s ------\n")