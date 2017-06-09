import collections
import os

import tensorflow as tf
import numpy as np
import gensim
import pickle

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


def input_generator(raw_data, batch_size, num_steps):
	data_len = len(raw_data)
	batch_len = data_len // batch_size
	raw_data = np.array(raw_data, dtype=np.int32)
	data = np.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
	epoch_size = (batch_len - 1) // num_steps
	for i in itertools.cycle(range(epoch_size)):
		x_batch = data[:, i*num_steps:(i+1)*num_steps]
		y_batch = data[:, i*num_steps+1:(i+1)*num_steps+1]
		yield x_batch, y_batch


def input_generator_continuous(raw_data, batch_size, num_steps):
	#batch_size = 1 # support other values? (faster evaluation)
	#num_steps = find_max_len(raw_data)
	data_len = len(raw_data)
	batch_len = data_len // batch_size
	raw_data = np.array(raw_data, dtype=np.int32)
	data = np.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
	epoch_size = batch_len  // (num_steps+1)
	for i in itertools.cycle(range(epoch_size)):
		slice_xy = data[:, i*(num_steps+1):(i+1)*(num_steps+1)]
		x_batch = slice_xy[:, 0:num_steps]
		y_batch = slice_xy[:, 1:num_steps+1]
		yield x_batch, y_batch


class InputGenerator(object):
	"""The input data."""
	def __init__(self, config, data, input_generator):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.generator = input_generator(data, batch_size, num_steps)

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
	state = session.run(model.initial_state)

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
	losses = []

	#state = session.run(model.initial_state)

	fetches = {
			"loss": model.loss,
			"correct_predictions": model.correct_predictions
	}

	accuracy = []

	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		feed_dict = {
			model.input_x : input_x,
			model.input_y : input_y,
			model.batch_size: input_x.shape[0],
		}
		results = session.run(fetches, feed_dict)
		loss = results["loss"]
		correct_predictions = results["correct_predictions"]
		
		inputs = input_x[0]
		
		not_pad = [elem != 0 for elem in inputs] # not pad

		relevant_index = max(loc for loc, val in enumerate(not_pad) if val == True) - 1 # previous word
		losses.append(loss[relevant_index])
		accuracy.append(correct_predictions[relevant_index])

	perplexity = np.exp(np.mean(losses))
	accuracy = np.mean(accuracy)  

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), {"perplexity": perplexity, "accuracy": accuracy}) # Write summary (CORPUS-WISE stats)

	return [perplexity, accuracy] 

def eval_last_word_cache(session, model, input_data, summary_writer=None):
	

	#state = session.run(model.initial_state)

	fetches = {
		"outputs": model.outputs,
		"logits": model.logits,
		#"loss": model.loss,
		#"correct_predictions": model.correct_predictions
	}

	losses = []
	accuracy = []
	
	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.get_batch()
		feed_dict = {
			model.input_x : input_x,
			model.input_y : input_y
		}
		results = session.run(fetches, feed_dict)
		rnn_outputs = results["outputs"] # list of np arrays of [1, hidden_size]
		logits = results["logits"] # np.array of [max_len, vocab_size]

		inputs = input_x[0]
		#correct_ids = input_y[0]
		
		not_pad = [elem != 0 for elem in inputs] # not pad
		last_word_index = max(loc for loc, val in enumerate(not_pad) if val == True)
		relevant_index = last_word_index - 1 # previous word

		# PARAMS

		theta = 0.3
		interpol = 0.7

		# Calculate LSTM probabilites manually
		relevant_logits = logits[relevant_index, :]
		word_probs = softmax(relevant_logits)


		# Calculate cache probabilities
		h_t = rnn_outputs[relevant_index]
		cache_logits = dict() # key: output word, value: logit

		for i in range(relevant_index): # words previous to the prediction
			pseudo_logit = np.exp( theta*np.sum(h_t*rnn_outputs[i]) )
			output_id = inputs[i+1] # or correct_ids[i]
			if output_id in cache_logits: 
				cache_logits[output_id] += pseudo_logit
			else:
				cache_logits[output_id] = pseudo_logit

		total_sum = sum(cache_logits.values())
		cache_probs = [float(val)/float(total_sum) for val in cache_logits.values()]
		cache_ids = cache_logits.keys()

		# Merge word and cache probabilities
		final_probs = (1-interpol)*np.array(word_probs)

		for i, output_id in enumerate(cache_ids):
			final_probs[output_id] += interpol*cache_probs[i]

		# Calculate loss
		true_output_id = inputs[last_word_index]
		loss = -np.log( final_probs[ true_output_id ] )
		losses.append(loss)

		# And accuracy
		predicted_id = np.argmax(final_probs)
		accuracy.append( predicted_id == true_output_id )

		if(step==0):
			print(rnn_outputs[0].shape)
			print(len(rnn_outputs))
			#print(logits)
			print(logits.shape)

			print("CHECK")
			print(len(relevant_logits))
			print(len(word_probs))


	perplexity = np.exp(np.mean(losses))
	accuracy = np.mean(accuracy)  

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), {"perplexity": perplexity, "accuracy": accuracy}) # Write summary (CORPUS-WISE stats)

	return [perplexity, accuracy] 

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