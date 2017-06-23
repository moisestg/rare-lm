import collections
import os

import tensorflow as tf
import numpy as np
import gensim
import pickle
import time
import math

import itertools


_EOS = "<eos>"
_UNK = "<unk>"
_PAD = "<pad>" # id: 0


## MISC ##

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def find_max_len(padded_raw_data):
	max_len = 0
	id = padded_raw_data.pop(0)
	while id != 0:
		max_len += 1
		id = padded_raw_data.pop(0)
	while id == 0:
		max_len +=1
		id = padded_raw_data.pop(0)
	return max_len

def relevant_index(row):
	return max(loc for loc, val in enumerate(row) if val != 0) - 1 # word previous to last


## GENERAL (DATA LOADING) ##

def get_vocab(train_path, vocab_size, tokenizer, use_unk=True):
	# Load if already exists
	pickle_path = os.path.split(train_path)[0]+"/preprocessed/dicts_vocSize"+str(vocab_size)+".pkl"
	if os.path.exists(pickle_path):
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	# Generate vocabulary
	word_counts = collections.Counter()
	with open(train_path, "r", encoding="utf-8") as f:
		for line in f:
			for word in tokenizer(line):
				if word in word_counts: 
					word_counts[word] += 1
				else:
					word_counts[word] = 1

	if use_unk:
		vocabulary = [tupl[0] for tupl in word_counts.most_common(vocab_size-2)]   
		word2id = {word: i for i, word in enumerate(vocabulary,2)}
		word2id[_PAD] = 0
		word2id[_UNK] = 1
	else:
		vocabulary = [tupl[0] for tupl in word_counts.most_common(vocab_size-1)]  
		word2id = {word: i for i, word in enumerate(vocabulary,1)}
		word2id[_PAD] = 0

	total_freq = sum(freq for freq in word_counts.values())
	kept_freq = sum(tupl[1] for tupl in word_counts.most_common(vocab_size)) # tupl -> (word, frequency)
	print("\n\n** Vocabulary of size "+str(vocab_size)+" covers "+str(round(kept_freq/total_freq*100, 2))+"% of the training words. **\n\n")

	# Generate id2word
	id2word = {v: k for k,v in word2id.items()}
	
	output_dir = os.path.split(train_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	with open(pickle_path, "wb") as f:
		pickle.dump([word2id, id2word], f)

	return [word2id, id2word]


def get_word_ids(data_path, word2id, tokenizer):
	# Load if already exists
	pickle_path = os.path.split(data_path)[0]+"/preprocessed/"+os.path.split(data_path)[1]+"_vocSize"+str(len(word2id))+".pkl"
	if os.path.exists(pickle_path):
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	data = []
	with open(data_path, "r", encoding="utf-8") as f:
		for line in f:
			for word in tokenizer(line):
			#for word in tokenizer(line.replace("\n", _EOS)):
				data.append(word2id[word] if word in word2id else word2id[_UNK])

	output_dir = os.path.split(data_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	with open(pickle_path, "wb") as f:
		pickle.dump(data, f)

	return data


def get_word_ids_padded(data_path, word2id, tokenizer):
	with open(data_path, "r", encoding="utf-8") as f:
		test_examples = f.readlines()

	test_examples = [tokenizer(example) for example in test_examples]
	max_seq_length = len(max(test_examples, key=len))

	data = []
	for example in test_examples:
		for word in example:
			data.append(word2id[word] if word in word2id else word2id[_UNK])
		for _ in range(max_seq_length-len(example)+1): # one extra pad, for the y
			data.append(word2id[_PAD])

	return data, max_seq_length


class input_generator(object):
	
	def __init__(self, raw_data, batch_size, num_steps):
		self.num_steps = num_steps
		data_len = len(raw_data)
		batch_len = data_len // batch_size
		raw_data = np.array(raw_data, dtype=np.int32)
		self.data = np.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
		self.epoch_size = (batch_len - 1) // num_steps

	def gen(self):
		data = self.data
		num_steps = self.num_steps
		epoch_size = self.epoch_size
		for i in itertools.cycle(range(epoch_size)):
			x_batch = data[:, i*num_steps:(i+1)*num_steps]
			y_batch = data[:, i*num_steps+1:(i+1)*num_steps+1]
			yield x_batch, y_batch

class input_generator_continuous(object):

	def __init__(self, raw_data, batch_size, num_steps):
		self.batch_size = batch_size
		self.num_steps = num_steps
		raw_data = np.array(raw_data, dtype=np.int32)
		self.data = np.reshape(raw_data, [-1, num_steps+1])
		self.epoch_size = math.ceil(self.data.shape[0]/batch_size)

	def gen(self):
		data = self.data
		num_steps = self.num_steps
		epoch_size = self.epoch_size
		batch_size = self.batch_size
		for i in itertools.cycle(range(epoch_size)):
			x_batch = data[i*batch_size:i*batch_size+batch_size, 0:num_steps]
			y_batch = data[i*batch_size:i*batch_size+batch_size, 1:num_steps+1]
			yield x_batch, y_batch


class InputGenerator(object):
	"""The input data."""
	def __init__(self, config, data, input_generator):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		generator = input_generator(data, batch_size, num_steps)
		self.epoch_size = generator.epoch_size
		self.generator = generator.gen()

	def get_batch(self):
		return next(self.generator)


## EVAL FUNCTIONS

def write_summary(summary_writer, current_step, values):
	list_values = []
	for key, value in values.items():
		list_values.append(tf.Summary.Value(tag=key, simple_value=value)) # TODO: Support other types of values (e.g. histogram)

	new_summ = tf.Summary()
	new_summ.value.extend(list_values)
	summary_writer.add_summary(new_summ, current_step)


def eval_epoch(session, model, input_data, summary_writer=None):
	costs = 0.0
	iters = 0
	accuracies = []
	state = session.run(model.initial_state, {model.epoch_size: input_data.epoch_size})

	fetches = {
			"cost": model.cost,
			"final_state": model.final_state,
			"accuracy": model.accuracy,
	}

	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		feed_dict = {
			model.input_x: input_x,
			model.input_y: input_y,
		}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h
		
		results = session.run(fetches, feed_dict)
		cost = results["cost"]
		state = results["final_state"]
		accuracy = results["accuracy"]

		costs += cost
		accuracies.append(accuracy)
		iters += input_data.num_steps
	
	perplexity = np.exp(costs / iters)
	accuracy = np.mean(accuracies)

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), {"perplexity": perplexity, "accuracy": accuracy}) # Write summary (CORPUS-WISE stats)	

	return [perplexity , accuracy]


def eval_last_word(session, model, input_data, summary_writer=None):

	fetches = {
			"loss": model.loss,
			"correct_predictions": model.correct_predictions
	}

	accuracies = np.array([])
	losses = np.array([])

	start_time = time.time()

	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		batch_size = input_x.shape[0]
		feed_dict = {
			model.input_x : input_x,
			model.input_y : input_y,
			model.batch_size: batch_size,
		}
		results = session.run(fetches, feed_dict)
		loss = results["loss"]
		correct_predictions = results["correct_predictions"]
		
		relevant_indexes = np.apply_along_axis(relevant_index, 1, input_x)
		loss = np.reshape(loss, (batch_size, -1))
		losses = np.append( losses, loss[np.arange(len(loss)), relevant_indexes] )
		correct_predictions = np.reshape(correct_predictions, (batch_size, -1))
		accuracies = np.append( accuracies, correct_predictions[np.arange(len(correct_predictions)), relevant_indexes] )

	perplexity = np.exp(np.mean(losses))
	accuracy = np.mean(accuracies)  

	print("Eval time: "+str(time.time()-start_time)+" s")

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), {"perplexity": perplexity, "accuracy": accuracy}) # Write summary (CORPUS-WISE stats)

	return [losses, accuracies]

def eval_last_word_detailed(session, model, input_data, id2word, pos):

	fetches = {
			"loss": model.loss,
			"correct_predictions": model.correct_predictions,
			"logits": model.logits,
	}

	accuracies = np.array([])
	losses = np.array([])
	ranks = np.array([])

	start_time = time.time()

	example_count = 0
	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		batch_size = input_x.shape[0]
		feed_dict = {
			model.input_x : input_x,
			model.input_y : input_y,
			model.batch_size: batch_size,
		}
		results = session.run(fetches, feed_dict)
		loss = results["loss"]
		correct_predictions = results["correct_predictions"]
		
		relevant_indexes = np.apply_along_axis(relevant_index, 1, input_x)
		loss = np.reshape(loss, (batch_size, -1))
		losses = np.append( losses, loss[np.arange(len(loss)), relevant_indexes] )
		correct_predictions = np.reshape(correct_predictions, (batch_size, -1))
		accuracies = np.append( accuracies, correct_predictions[np.arange(len(correct_predictions)), relevant_indexes] )

		# DETAILED STUFF PER EXAMPLE
		logits = results["logits"] # np.array of [batch_size*max_len, vocab_size]
		vocab_size = logits.shape[1]
		logits = np.reshape(logits, (input_x.shape[0], -1, vocab_size)) # [batch_size, max_len, vocab_size]
		with open("detailed_output_"+str(int(start_time))+".txt", "a") as f:
			for b in range(batch_size):
				f.write("* EXAMPLE "+str(example_count)+":\n")
				# Target word info
				target_word_id = input_y[b, relevant_indexes[b]]
				f.write("Target word: "+id2word[ target_word_id ]+" | PoS tag: "+pos[example_count]+"\n")
				# Top k predictions
				relevant_logits = logits[b, relevant_indexes[b], :] # [vocab_size]
				ordered_indexes = relevant_logits.argsort() # from less to more
				topk_indexes = ordered_indexes[-10:][::-1]
				f.write("Top 10 predictions:")
				for index in topk_indexes:
					f.write(" "+id2word[index])
				f.write("\n")
				# Target word rank
				target_word_rank = vocab_size - np.where(ordered_indexes == target_word_id)[0][0]
				ranks = np.append(ranks, target_word_rank)
				f.write("Target word rank: "+str(target_word_rank)+"\n")
				# Word perplexities: word/perplexity when predicting that word (the final prediction might have been different than the target)
				f.write("Word perplexities (word/perplexity):\n")
				for i in range(relevant_indexes[b]+2): # until the last word
					f.write(id2word[ input_x[b, i] ]) # word
					if i>0:
						f.write("/"+str(round(np.exp(loss[b, i-1]), 2))+" ")
					else:
						f.write(" ")
				f.write("\n\n")
				example_count += 1

	perplexity = np.exp(np.mean(losses))
	accuracy = np.mean(accuracies)
	rank = np.median(ranks)
	with open("detailed_output_"+str(int(start_time))+".txt", "a") as f:
		f.write("Average (target word) perplexity: "+str(perplexity)+" | Average (target word) accuracy: "+str(accuracy)
			+" | Median (target word) rank: "+str(rank))

	print("Detailed eval time: "+str(time.time()-start_time)+" s")

# TODO: Avoid for loops ?
def eval_last_word_cache(session, model, input_data, summary_writer=None):

	fetches = {
		"outputs": model.outputs,
		"logits": model.logits,
	}

	accuracies = np.array([])
	losses = np.array([])

	start_time = time.time()
	
	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		batch_size = input_x.shape[0]
		feed_dict = {
			model.input_x : input_x,
			model.input_y : input_y,
			model.batch_size: batch_size,
		}
		results = session.run(fetches, feed_dict)
		rnn_outputs = results["outputs"] # list of "max_len" np arrays of [batch_size, hidden_size]
		logits = results["logits"] # np.array of [batch_size*max_len, vocab_size]
		logits = np.reshape(logits, (input_x.shape[0], -1, logits.shape[1])) # [batch_size, max_len, vocab_size]

		relevant_indexes = np.apply_along_axis(relevant_index, 1, input_x) # [batch_size]
		last_word_indexes = relevant_indexes + 1

		# PARAMS
		# TODO: Pass them through the FLAGS or smthng
		theta = 0.3
		interpol = 0.7

		# Calculate LSTM probabilites manually
		relevant_logits = logits[np.arange(len(logits)), relevant_indexes, :] # [batch_size, vocab_size]
		word_probs = np.apply_along_axis(softmax, 1, relevant_logits) # [batch_size, vocab_size]

		for b in range(input_x.shape[0]): # batch_size		

			# Calculate cache probabilities
			h_t = rnn_outputs[relevant_indexes[b]][b,:]
			cache_logits = dict() # key: output word, value: logit

			for i in range(relevant_indexes[b]): # words previous to the prediction
				pseudo_logit = np.exp( theta*np.sum( h_t*rnn_outputs[i][b,:]) )
				output_id = input_x[b,:][i+1] # or correct_ids[i]
				if output_id in cache_logits: 
					cache_logits[output_id] += pseudo_logit
				else:
					cache_logits[output_id] = pseudo_logit

			total_sum = sum(cache_logits.values())
			cache_probs = [float(val)/float(total_sum) for val in cache_logits.values()]
			cache_ids = cache_logits.keys()

			# Merge word (RNN) and cache probabilities
			final_probs = (1-interpol)*word_probs[b,:] # [vocab_size]

			for i, output_id in enumerate(cache_ids):
				final_probs[output_id] += interpol*cache_probs[i]

			# Calculate loss
			true_output_id = input_x[b, last_word_indexes[b]]
			loss = -np.log( final_probs[ true_output_id ] )
			losses = np.append(losses, loss)

			# And accuracy
			predicted_id = np.argmax(final_probs)
			accuracies = np.append( accuracies, predicted_id == true_output_id )

	perplexity = np.exp(np.mean(losses))
	accuracy = np.mean(accuracies) 

	print("Eval time: "+str(time.time()-start_time)+" s") 

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), {"perplexity": perplexity, "accuracy": accuracy}) # Write summary (CORPUS-WISE stats)

	return [losses, accuracies]

def eval_last_word_cache_detailed(session, model, input_data, id2word, pos):

	fetches = {
		"outputs": model.outputs,
		"logits": model.logits,
	}

	accuracies = np.array([])
	losses = np.array([])
	ranks = np.array([])

	start_time = time.time()
	
	example_count = 0
	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		batch_size = input_x.shape[0]
		feed_dict = {
			model.input_x : input_x,
			model.input_y : input_y,
			model.batch_size: batch_size,
		}
		results = session.run(fetches, feed_dict)
		rnn_outputs = results["outputs"] # list of "max_len" np arrays of [batch_size, hidden_size]
		logits = results["logits"] # np.array of [batch_size*max_len, vocab_size]
		vocab_size = logits.shape[1]
		logits = np.reshape(logits, (input_x.shape[0], -1, vocab_size)) # [batch_size, max_len, vocab_size]

		relevant_indexes = np.apply_along_axis(relevant_index, 1, input_x) # [batch_size]
		last_word_indexes = relevant_indexes + 1

		# PARAMS
		# TODO: Pass them through the FLAGS or smthng
		theta = 0.3
		interpol = 0.7

		# Calculate LSTM probabilites manually
		relevant_logits = logits[np.arange(len(logits)), relevant_indexes, :] # [batch_size, vocab_size]
		word_probs = np.apply_along_axis(softmax, 1, relevant_logits) # [batch_size, vocab_size]

		# DETAILED STUFF PER EXAMPLE
		all_word_probs = np.apply_along_axis(softmax, 2, logits) # [batch_size, max_len, vocab_size]
		with open("detailed_output_"+str(int(start_time))+".txt", "a") as f:
			for b in range(input_x.shape[0]): # batch_size
				f.write("* EXAMPLE "+str(example_count)+":\n")
				# Target word info
				target_word_id = input_y[b, relevant_indexes[b]]
				f.write("Target word: "+id2word[ target_word_id ]+" | PoS tag: "+pos[example_count]+"\n")
				cache = [] # key: output word, value: previous hidden state	
				# Update all probs by mixing word + cache	
				for i in range(1, relevant_indexes[b]+2): # +1 ???? until last word
					# Calculate cache probabilities
					h_t = rnn_outputs[i][b,:]
					#pseudo_logit = np.exp( theta*np.sum( h_t*rnn_outputs[i-1][b,:] ) )
					output_id = input_x[b,:][i]
					cache.append((output_id, rnn_outputs[i-1][b,:]))
					cache_logits = [(tupl[0], np.exp(theta*np.sum(h_t*tupl[1]))) for tupl in cache]
					total_sum = np.sum([tupl[1] for tupl in cache_logits])
					cache_probs = [float(tupl[1])/float(total_sum) for tupl in cache_logits]
					cache_ids = [tupl[0] for tupl in cache_logits]

					# Merge word (RNN) and cache probabilities
					all_word_probs[b, i, :] = (1-interpol)*all_word_probs[b, i, :] # [vocab_size]
					for j, output_id in enumerate(cache_ids):
						all_word_probs[b, i, output_id] += interpol*cache_probs[j]
				# Top k predictions
				relevant_probs = all_word_probs[b, relevant_indexes[b], :] # [vocab_size]
				ordered_indexes = relevant_probs.argsort() # from less to more
				topk_indexes = ordered_indexes[-10:][::-1]
				f.write("Top 10 predictions:")
				for index in topk_indexes:
					f.write(" "+id2word[index])
				f.write("\n")
				# Target word rank
				target_word_rank = vocab_size - np.where(ordered_indexes == target_word_id)[0][0]
				ranks = np.append(ranks, target_word_rank)
				f.write("Target word rank: "+str(target_word_rank)+"\n")
				# Word perplexities: word/perplexity when predicting that word (the final prediction might have been different than the target)
				f.write("Word perplexities (word/perplexity):\n")
				for i in range(relevant_indexes[b]+2): # until the last word
					f.write(id2word[ input_x[b, i] ]) # word
					if i>0:
						f.write("/"+str( round(np.exp(-np.log( all_word_probs[b, i-1, target_word_id] )) , 2))+" ")
					else:
						f.write(" ")
				f.write("\n\n")
				example_count += 1

				# Calculate loss
				loss = -np.log( all_word_probs[b, relevant_indexes[b], target_word_id] )
				losses = np.append(losses, loss)

				# And accuracy
				predicted_id = np.argmax(all_word_probs[b, relevant_indexes[b], :])
				accuracies = np.append( accuracies, predicted_id == target_word_id )

	perplexity = np.exp(np.mean(losses))
	accuracy = np.mean(accuracies) 
	rank = np.median(ranks)
	with open("detailed_output_"+str(int(start_time))+".txt", "a") as f:
		f.write("Average (target word) perplexity: "+str(perplexity)+" | Average (target word) accuracy: "+str(accuracy)
				+" | Median (target word) rank: "+str(rank))

	print("Detailed eval time: "+str(time.time()-start_time)+" s") 


## LAMBADA DATASET

class LambadaDataset(object):

	def tokenizer(self, line):
		return line.strip().split(" ")

	def get_vocab(self, train_path, vocab_size):
		return get_vocab(train_path, vocab_size, self.tokenizer, use_unk=True)

	def get_train_data(self, data_path, word2id): # get_train_data() ??
		return get_word_ids(data_path, word2id, self.tokenizer)

	def get_dev_data(self, data_path, word2id):
		return get_word_ids_padded(data_path, word2id, self.tokenizer)

	def get_test_data(self, data_path, word2id):
		return get_word_ids_padded(data_path, word2id, self.tokenizer)

	def get_train_batch_generator(self, config, data):
		return InputGenerator(config, data, input_generator)

	def get_dev_batch_generator(self, config, data):
		return InputGenerator(config, data, input_generator_continuous)

	def get_test_batch_generator(self, config, data):
		return InputGenerator(config, data, input_generator_continuous)

	def eval_dev(self, session, model, input_data, summary_writer=None):
		return eval_last_word(session, model, input_data, summary_writer)

	def eval_test(self, session, model, input_data, summary_writer=None):
		return eval_last_word(session, model, input_data, summary_writer)

	def eval_detailed(self, session, model, input_data, id2word, pos):
		return eval_last_word_detailed(session, model, input_data, id2word, pos)


# PENN TREE BANK (PTB) DATASET

class PTBDataset(object):

	def tokenizer(self, line):
		return line.replace("\n", " "+_EOS).strip().split(" ")

	def get_vocab(self, train_path, vocab_size):
		return get_vocab(train_path, vocab_size, self.tokenizer, use_unk=False)

	def get_train_data(self, data_path, word2id):
		return get_word_ids(data_path, word2id, self.tokenizer)

	def get_dev_data(self, data_path, word2id):
		return get_word_ids(data_path, word2id, self.tokenizer)

	def get_test_data(self, data_path, word2id):
		return get_word_ids(data_path, word2id, self.tokenizer)

	def get_train_batch_generator(self, config, data):
		return InputGenerator(config, data, input_generator=input_generator)

	def get_dev_batch_generator(self, config, data):
		return InputGenerator(config, data, input_generator=input_generator)

	def get_test_batch_generator(self, config, data):
		return InputGenerator(config, data, input_generator=input_generator)

	def eval_dev(self, session, model, input_data, summary_writer=None):
		return eval_epoch(session, model, input_data, summary_writer)

	def eval_test(self, session, model, input_data, summary_writer=None):
		return eval_epoch(session, model, input_data, summary_writer)


## WORD EMBEDDINGS ##

def get_word2vec(train_path, vector_dim, word2id):
	# Load if already exists
	pickle_path = os.path.split(train_path)[0]+"/preprocessed/word2vec_dim"+str(vector_dim)+"_vocSize"+str(len(word2id))+".pkl"
	if os.path.exists(pickle_path):
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	print("Training word2vec...")
	# Train sentences generator
	class Sentence_generator(object):
		def __init__(self, train_path):
			self.train_path = train_path
		
		def __iter__(self):
			for line in open(self.train_path, "r", encoding="utf-8"):
				yield tokenizer(line)
	# TODO: Include EOS ??
	sentence_iterator = Sentence_generator(train_path)
	model = gensim.models.Word2Vec(sentence_iterator, size=vector_dim)

	vectors = np.empty([len(word2id), vector_dim], dtype='float32')
	for word in word2id.keys():
		# CHECK IF WORD EXISTS
		if word in model.wv:
			vectors[word2id[word],:] = model.wv[word]
		else:
			vectors[word2id[word],:] = np.random.randn(vector_dim)

	output_dir = os.path.split(train_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	with open(pickle_path, "wb") as f:
		pickle.dump(vectors, f)

	return vectors