import tensorflow as tf
import argparse
from binaryClassifier import BinaryClassifier
import data_utils_old
import data_utils
import numpy as np
import time
import os
import sys
import collections

## PARAMETERS ##
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument("--train_path", type=str, default="capitalized_train", help="Path to the training data") #/home/moises/thesis/lambada/lambada-dataset/train-novels/
parser.add_argument("--dev_path", type=str, default="capitalized_dev", help="Path to the dev data") #/home/moises/thesis/lambada/lambada-dataset/dev/
parser.add_argument("--save_path", type=str, default="./runs/", help="Path to save the model's checkpoints and summaries")

parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden & cell state")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate of the optimizer")

# Training parameters 
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--evaluate_every", type=int, default=10000, help="Evaluate model on dev set after this many steps")
parser.add_argument("--checkpoint_every", type=int, default=10000, help="Save model after this many steps ")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")
parser.add_argument("--restore_path", type=str, default=None, help="Path to the model to resume the training (default: None)")

FLAGS, _ = parser.parse_known_args()

print("\n- Parameters:")
flags_list = sorted(vars(FLAGS))
for flag in flags_list:
	print("  --"+flag+"="+str(getattr(FLAGS, flag)))

FLAGS.batch_size=64
FLAGS.num_steps=35
train_path="../lambada-dataset/capitalized/capitalized_train.txt"
vocab_size=63687
dataset = data_utils_old.LambadaDataset()
word2id, id2word = dataset.get_vocab(train_path, vocab_size)
train_data = dataset.get_train_data(train_path, word2id)
train_input = dataset.get_train_batch_generator(config=FLAGS, data=train_data)

# Define graph
with tf.Graph().as_default():
	# Initialize the model 
	#words_generator = data_utils.debug_input(FLAGS.train_path)
	train_input = data_utils.BatchGenerator(FLAGS.train_path)
	with tf.variable_scope("model", reuse=None):
		model_train = BinaryClassifier(is_training=True, config=FLAGS)

	valid_input = data_utils.BatchGenerator(FLAGS.dev_path) # 5% of capitalized_train
	with tf.variable_scope("model", reuse=True):
		model_valid = BinaryClassifier(is_training=False, config=FLAGS)

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

			# Train epoch variables
			start_time = time.time()
			iters = 0
			
			fetches = {
				"loss": model_train.loss,
				"predictions": model_train.predictions,
				"train_op" : model_train.train_op,
			}

			# Iterate through all batches (one epoch)
			for step in range(train_input.epoch_size): 
				
				_, words_y = train_input.get_batch()
				input_x, input_y = train_input.getBatch()
				feed_dict = {
					model_train.input_x: input_x,
					model_train.input_y: input_y,
				}
				
				# Run batch
				results = session.run(fetches, feed_dict)
				predictions = results["predictions"]
				loss = results["loss"]

				iters += input_x.shape[0] # batch_size
				current_step = sv.global_step.eval(session) 

				#next_words = next(words_generator)
				# Write train summary 
				if step % 100 == 0:
					fscoreName, fscoreNoName, auc = data_utils.getStats(input_y, predictions)
					print("Step: %i: fscoreName: %.3f, fscoreNoName: %.3f, auc: %.3f, loss: %.3f, Speed: %.0f Hz" %
								(current_step, fscoreName, fscoreNoName, auc, loss,
								 iters / (time.time() - start_time)))
					data_utils.write_summary(train_summary_writer, current_step, 
						{"fscoreName":fscoreName, "fscoreNoName":fscoreNoName, "auc":auc, "loss":loss})
					# DEBUG INPUT
					print("Accuracy: "+str(sum(input_y==predictions)/len(input_y)*100))
					print([id2word[i] for i in words_y.reshape(-1)[predictions==1]])

				# Eval on dev set
				if current_step % FLAGS.evaluate_every == 0:
					data_utils.eval_epoch(session, model_valid, valid_input, dev_summary_writer)

				# Checkpoint model
				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(session, checkpoint_prefix, global_step=current_step)
					print("\n** Saved model checkpoint to {} **\n".format(path))

			print("\n\n----- Last epoch took a total of: "+str(time.time() - start_time)+" s ------\n")