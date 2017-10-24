import tensorflow as tf
import numpy as np
import argparse
import os
import time

from psm_lm import PointerSentinelMixtureLM
import lambada_utils as utils

## PARAMETERS ##
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Data loading parameters
parser.add_argument("--train_path", type=str, help="Path to the training data")
parser.add_argument("--dev_path", type=str, help="Path to the dev data")
parser.add_argument("--save_path", type=str, default="./runs/", help="Path to save the model's checkpoints and summaries")

# Model parameters
parser.add_argument("--pretrained_emb", type=str, default=None, help="Pretrained vectors to initialize the embedding matrix")
parser.add_argument("--emb_size", type=int, default=200, help="Dimensionality of word embeddings")
parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary")
parser.add_argument("--num_layers", type=int, default=1, help="Number of recurrent layers")
parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden & cell state")
parser.add_argument("--projection_size", type=int, default=None, help="Size of the output projection of the hidden states")
parser.add_argument("--num_steps_train", type=int, default=35, help="Number of unrolled steps (train)")
parser.add_argument("--num_steps_eval", type=int, default=35, help="Number of unrolled steps (eval)")
parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate of the optimizer")
parser.add_argument("--learning_rate_decay", type=float, default=None, help="Learning rate decay")
parser.add_argument("--emb_keep_prob", type=float, default=1.0, help="Dropout embedding keep probability")
parser.add_argument("--input_keep_prob", type=float, default=1.0, help="Dropout input keep probability")
parser.add_argument("--output_keep_prob", type=float, default=1.0, help="Dropout output keep probability")
parser.add_argument("--state_keep_prob", type=float, default=1.0, help="Dropout state keep probability")
parser.add_argument("--l2_reg", type=float, default=0.01, help="Weight for L2 regularization")
parser.add_argument("--clip_norm", type=float, default=5.0, help="Norm value to clip the gradients")
parser.add_argument("--optimizer_algo", type=str, default="adam", help="Optimization algorithm")
parser.add_argument("--attention_length", type=int, default=100, help="Length of the pointer cache")

# Training parameters
parser.add_argument("--batch_size_train", type=int, default=64, help="Batch size (train)") 
parser.add_argument("--batch_size_eval", type=int, default=64, help="Batch size (eval)") 
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--evaluate_every", type=int, default=1000, help="Evaluate model on dev set after this many steps")
parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save model after this many steps")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store")
parser.add_argument("--restore_path", type=str, default=None, help="Path to the model to resume the training")

config, _ = parser.parse_known_args()

print("\n- Parameters:")
config_list = sorted(vars(config))
for param in config_list:
	print("  --"+param+"="+str(getattr(config, param)))

## MAIN ##

# Load data
word2id, id2word = utils.get_vocab(config.train_path, config.vocab_size)
lm_trainData = utils.SlidingGenerator(config.train_path, word2id, config.batch_size_train, config.num_steps_train, shuffle=True)
lm_validData = utils.SlidingGenerator(config.dev_path, word2id, config.batch_size_eval, config.num_steps_eval)

# Load pretrained embeddings (if any)
if config.pretrained_emb == "word2vec":
	pretrained_emb = utils.get_word2vec(config.train_path, config.emb_size, word2id)
else:
	pretrained_emb = None

with tf.Graph().as_default():

	# Initialize the model
	initializer = tf.random_uniform_initializer(-.1, .1) # default variable initializer

	with tf.variable_scope("model", reuse=None, initializer=initializer):
		model_train = PointerSentinelMixtureLM(config=config, is_training=True, pretrained_emb=pretrained_emb)

	with tf.variable_scope("model", reuse=True, initializer=initializer):
		model_valid = PointerSentinelMixtureLM(config=config, is_training=False, pretrained_emb=pretrained_emb)

	# Define saver to checkpoint the model
	out_path = config.save_path + str(int(time.time()))
	checkpoint_path = os.path.abspath(os.path.join(out_path, "checkpoints"))
	checkpoint_prefix = os.path.join(checkpoint_path, "model")
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

	# Create session
	sv = tf.train.Supervisor(logdir=None)
	with sv.managed_session() as session:
		## TRAIN LOOP ##

		# Restore previous model (if any) to resume training
		if config.restore_path is not None:
			saver.restore(session, config.restore_path)
			print("\n** Model restored from: "+config.restore_path+" **\n")
			session.run(tf.global_variables())

		# Define sumary writers
		train_summary_path = os.path.join(out_path, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_path, session.graph)
		dev_summary_path = os.path.join(out_path, "summaries", "dev")
		dev_summary_writer = tf.summary.FileWriter(dev_summary_path, session.graph)

		prev_valid_perp = float("Inf")
		current_lr = config.learning_rate

		print("\n\n\n*** START TRAINING ***\n\n")

		for epoch in range(config.num_epochs):

			print("\nStarting epoch: %d" % (epoch + 1))

			# Epoch variables
			epoch_start_time = start_time = time.time()
			iters = 0
			states = session.run(model_train.initial_states)
			attention_states = session.run(model_train.initial_attention_states)
			attention_ids = session.run(model_train.initial_attention_ids)

			fetches_small = {
				"final_states": model_train.final_states,
				"final_attention_states": model_train.final_attention_states,
				"final_attention_ids": model_train.final_attention_ids,
				"train_op": model_train.train_op,
			}

			fetches = {
				"lm_loss": model_train.lm_loss,
				"reg_loss": model_train.reg_loss,
				"loss": model_train.loss,
				"perplexity_name": model_train.perplexity_name,
				"perplexity_notName": model_train.perplexity_notName,
				"perplexity": model_train.perplexity,
				"jsd": model_train.jsd,
				"pointer_loss": model_train.pointer_loss,
				"gate_names": model_train.gate_names,
				"gate_notNames": model_train.gate_notNames,
				"final_states": model_train.final_states,
				"final_attention_states": model_train.final_attention_states,
				"final_attention_ids": model_train.final_attention_ids,
				"train_op": model_train.train_op,
				"attention_scores": model_train.attention_scores
			}

			# Iterate through all batches (one epoch)
			for batch in range(lm_trainData.epoch_size): 
				
				lm_x, lm_y, switch_y = next(lm_trainData.generator)
				#print(attention_ids)
				feed_dict = {
					model_train.lm_x: lm_x,
					model_train.lm_y: lm_y,
					model_train.switch_y: switch_y.reshape(-1),
					model_train.initial_attention_states: attention_states,
					model_train.initial_attention_ids: attention_ids,
				}

				# Feed previous state
				for i, (c, h) in enumerate(model_train.initial_states):
					feed_dict[c] = states[i].c
					feed_dict[h] = states[i].h
				
				# Run batch
				global_step = sv.global_step.eval(session)
				
				if global_step % 100 == 0:
					results = session.run(fetches, feed_dict)
				else:
					results = session.run(fetches_small, feed_dict)

				states = results["final_states"] # propagate cell states between batches
				attention_states = results["final_attention_states"]
				attention_ids = results["final_attention_ids"]

				iters += lm_trainData.num_steps # *batch_size (done on the fly to avoid overflow)

				# Eval on dev set
				if global_step % config.evaluate_every == 0:
					valid_perp = utils.eval_dev(session, model_valid, lm_validData, dev_summary_writer)

				# Checkpoint model
				if global_step % config.checkpoint_every == 0:
					path = saver.save(session, checkpoint_prefix, global_step=global_step)
					print("\n** Saved model checkpoint to {} **\n".format(path))

				# Print some info and write train summary
				if global_step % 100 == 0:

					lm_loss = results["lm_loss"]
					reg_loss = results["reg_loss"]
					loss = results["loss"]
					perplexity_name = np.exp(results["perplexity_name"])
					perplexity_notName = np.exp(results["perplexity_notName"])
					perplexity = np.exp(results["perplexity"]) # word perplexity
					jsd = results["jsd"]
					pointer_loss = results["pointer_loss"]
					gate_names = results["gate_names"]
					gate_notNames = results["gate_notNames"]
 
					utils.write_summary(train_summary_writer, global_step, {"perplexity": perplexity,
						"perplexity_name": perplexity_name, "perplexity_notName": perplexity_notName,
						"loss": loss, "lm_loss": lm_loss, "reg_loss": reg_loss, "jsd": jsd,
						"gate_names": gate_names, "gate_notNames": gate_notNames,
						"pointer_loss": pointer_loss})

					print("Step: %i: Perplexity: %.3f, Speed: %.0f wps" %
						(global_step, perplexity, iters * lm_trainData.batch_size / (time.time() - start_time)))

					#print(results["attention_scores"])

					start_time = time.time() # reset
					iters = 0

			if config.learning_rate_decay and valid_perp > prev_valid_perp:
				current_lr *= config.learning_rate_decay
				print("\n\n ** Decreasing learning rate to "+str(current_lr)+" ** \n\n")
				model_train.assign_lr(session, current_lr)
				prev_valid_perp = valid_perp

			print("\n\n*** Last epoch took a total of: "+str(time.time() - epoch_start_time)+" s ***\n")