import tensorflow as tf
import numpy as np
import gensim
import pickle
import os
import time
import collections
import itertools

from itertools import groupby
import random

_PAD = "<pad>"
_UNK = "<unk>"
_BOC = "<boc>"
#_EOS = "<eos>"


def softmax_stable(logits):
	logits = logits-tf.expand_dims(tf.reduce_max(logits, axis=-1), 1)
	return tf.nn.softmax(logits)


def softmax_stable_target(logits, target_indices):
	logits = logits-tf.expand_dims(tf.reduce_max(logits, axis=-1), 1)
	logits_exp = tf.exp(logits)
	logits_targets = tf.gather_nd(logits_exp, indices=target_indices)
	sums = tf.reduce_sum(logits_exp, axis=-1)
	return logits_targets/sums


_EPSILON=1e-30
def log_stable(probs):
	return tf.log(probs+_EPSILON)


def jsd(p, q):
	m = 0.5 * (p + q)
	return 0.5 * ( tf.reduce_sum(p * log_stable(p / m)) + tf.reduce_sum(q * log_stable(q / m)) )


def tokenizer(line):
	return line.strip().split()


def get_vocab(train_path, vocab_size):
	# Load if already exists
	pickle_path = os.path.split(train_path)[0]+"/preprocessed/dicts_vocSize"+str(vocab_size)+".pkl"
	if os.path.exists(pickle_path):
		print("\nRestoring vocab from: "+pickle_path)
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

	vocabulary = set([tupl[0] for tupl in word_counts.most_common(vocab_size+1)])
	vocabulary.remove(_BOC)

	# Generate ids
	word2id = {word: i for i, word in enumerate(vocabulary, 3)}
	word2id[_PAD] = 0
	word2id[_UNK] = 1
	word2id[_BOC] = 2

	# Generate id2word
	id2word = {v: k for k,v in word2id.items()}
	
	output_dir = os.path.split(train_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(pickle_path, "wb") as f:
		pickle.dump([word2id, id2word], f)

	return [word2id, id2word]


def get_wordIDs(data_path, word2id):
	# Load if already exists
	pickle_path = os.path.split(data_path)[0]+"/preprocessed/"+os.path.split(data_path)[1]+"_vocSize"+str(len(word2id))+".pkl"
	if os.path.exists(pickle_path):
		print("Restoring word IDs from: "+pickle_path)
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	ids = []
	with open(data_path, "r", encoding="utf-8") as f:
		for line in f:
			for word in tokenizer(line):
				if word == _BOC: # or word == _EOS
					ids.append(-1)
				if word in word2id:
					ids.append(word2id[word])
				else:
					ids.append(word2id[_UNK])

	idsChunks = [list(group) for k, group in groupby(ids, lambda x: x == -1) if not k]

	output_dir = os.path.split(data_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	with open(pickle_path, "wb") as f:
		pickle.dump(idsChunks, f)

	return idsChunks


def get_switchData(data_path):
	# Load if already exists
	pickle_path = os.path.split(data_path)[0]+"/preprocessed/"+os.path.split(data_path)[1]+".nameTags"+".pkl"
	if os.path.exists(pickle_path):
		print("Restoring switch data from: "+pickle_path)
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	# Load names list
	with open("./names_list_lower.pkl", "rb") as f:
		names_list = pickle.load(f)

	switchData = []
	with open(data_path, "r", encoding="utf-8") as f:
		for line in f:
			for word in line.strip().split():
				if word == _BOC: # or word == _EOS
					switchData.append(-1)
				if word in names_list:
					switchData.append(1)
				else:
					switchData.append(0)

	switchChunks = [list(group) for k, group in groupby(switchData, lambda x: x == -1) if not k]

	output_dir = os.path.split(data_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	with open(pickle_path, "wb") as f:
		pickle.dump(switchChunks, f)

	return switchChunks

"""
def get_switchData(data_path):
	# Load if already exists
	pickle_path = os.path.split(data_path)[0]+"/preprocessed/"+os.path.split(data_path)[1]+".tags"+".pkl"
	if os.path.exists(pickle_path):
		print("Restoring switch data from: "+pickle_path)
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	switchData = []
	with open(data_path+".tags", "r", encoding="utf-8") as f:
		for line in f:
			tag = line.strip():
			if tag == "NNP" or tag == "NNPS":
				switchData.append(1)
			else:
				switchData.append(0)

	output_dir = os.path.split(data_path)[0]+"/preprocessed/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	with open(pickle_path, "wb") as f:
		pickle.dump(switchData, f)

	return switchData
"""

class SlidingGenerator(object):

	def __init__(self, data_path, word2id, batch_size, num_steps, shuffle=False):
		self.data_path = data_path
		self.word2id = word2id
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.lmData, self.switchData, self.epoch_size = self.getData(shuffle)
		self.generator = self.gen()

	def getData(self, shuffle):
		idsChunks = get_wordIDs(self.data_path, self.word2id)
		switchChunks = get_switchData(self.data_path)
		# Shuffle
		if shuffle:
			both = list(zip(idsChunks, switchChunks))
			random.shuffle(both)
			idsChunks, switchChunks = zip(*both)
		# Flatten
		ids = np.array([item for chunk in idsChunks for item in chunk], dtype=np.int32)
		switch = np.array([item for chunk in switchChunks for item in chunk], dtype=np.int32)
		# Shape
		batch_size = self.batch_size
		num_steps = self.num_steps
		data_len = len(ids)
		batch_len = data_len // batch_size
		epoch_size = (batch_len - 1) // num_steps
		lmData = np.reshape(ids[0 : batch_size * batch_len], [batch_size, batch_len])
		switchData = np.reshape(switch[0 : batch_size * batch_len], [batch_size, batch_len])
		return lmData, switchData, epoch_size

	def gen(self):
		epoch_size = self.epoch_size
		num_steps = self.num_steps

		for i in itertools.cycle(range(epoch_size)):
			lm_x = self.lmData[:, i*num_steps:(i+1)*num_steps]
			lm_y = self.lmData[:, i*num_steps+1:(i+1)*num_steps+1]
			switch_y = self.switchData[:, i*num_steps+1:(i+1)*num_steps+1]
			yield lm_x, lm_y, switch_y
			
			if self.shuffle and i == epoch_size-1:
				self.lmData, self.switchData, _ = self.getData(self.shuffle)


def eval_dev(session, model, lm_data, summary_writer=None):

	states = session.run(model.initial_states)
	attention_states = session.run(model.initial_attention_states)
	attention_ids = session.run(model.initial_attention_ids)

	fetches = {
		"final_states": model.final_states,
		"final_attention_states": model.final_attention_states,
		"final_attention_ids": model.final_attention_ids,
		"lm_loss": model.lm_loss,
		"reg_loss": model.reg_loss,
		"loss": model.loss,
		"perplexity_name": model.perplexity_name,
		"perplexity_notName": model.perplexity_notName,
		"perplexity": model.perplexity,
		"jsd": model.jsd,
		"pointer_loss": model.pointer_loss,
		"gate_names": model.gate_names,
		"gate_notNames": model.gate_notNames,
	}

	start_time = time.time()

	lm_loss = []
	reg_loss = []
	loss = []
	perplexity_name = []
	perplexity_notName = []
	perplexity = []
	jsd = []
	pointer_loss = []
	gate_names = []
	gate_notNames = []

	for step in range(lm_data.epoch_size):
		lm_x, lm_y, switch_y = next(lm_data.generator)
		feed_dict = {
			model.lm_x: lm_x,
			model.lm_y: lm_y,
			model.switch_y: switch_y.reshape(-1),
			model.initial_attention_states: attention_states,
			model.initial_attention_ids: attention_ids,
		}

		for i, (c, h) in enumerate(model.initial_states):
			feed_dict[c] = states[i].c
			feed_dict[h] = states[i].h

		results = session.run(fetches, feed_dict)
		states = results["final_states"]
		attention_states = results["final_attention_states"]
		attention_ids = results["final_attention_ids"]
		lm_loss.append( results["lm_loss"] )
		reg_loss.append( results["reg_loss"] )
		loss.append( results["loss"] )
		perplexity_name.append( results["perplexity_name"] )
		perplexity_notName.append( results["perplexity_notName"] )
		perplexity.append( results["perplexity"] )
		jsd.append( results["jsd"] )
		pointer_loss.append( results["pointer_loss"] )
		gate_names.append( results["gate_names"] )
		gate_notNames.append( results["gate_notNames"] )

	lm_loss = np.mean(lm_loss)
	reg_loss = np.mean(reg_loss)
	loss = np.mean(loss)
	perplexity_name = np.exp(np.mean(perplexity_name))
	perplexity_notName = np.exp(np.mean(perplexity_notName))
	perplexity = np.exp(np.mean(perplexity))
	jsd = np.mean(jsd)
	pointer_loss = np.mean(pointer_loss)
	gate_names = np.mean(gate_names)
	gate_notNames = np.mean(gate_notNames)

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), {"perplexity": perplexity,
			"perplexity_name": perplexity_name, "perplexity_notName": perplexity_notName,
			"loss": loss, "lm_loss": lm_loss, "reg_loss": reg_loss, "jsd": jsd,
			"gate_names": gate_names, "gate_notNames": gate_notNames,
			"pointer_loss": pointer_loss}) # Write summary (CORPUS-WISE stats) 
	
	print("\n** Valid Perplexity: %.3f **" % (perplexity))
	print("Eval time: "+str(time.time()-start_time)+" s")

	return perplexity


def eval_test(session, model, lm_data, switch_data, id2word):

	outFile_name = str(int(time.time()))+"_testOutput.txt"
	batch_size = model.batch_size
	num_steps = model.num_steps

	states = session.run(model.initial_states)

	fetches = {
		"final_states": model.final_states,
		"loss": model.loss,
		"perplexity_name": model.perplexity_name,
		"perplexity_notName": model.perplexity_notName,
		"perplexity": model.perplexity,
		"perplexities": model.perplexities,
		"rank_predictions": model.rank_predictions,
		"top_predictions": model.top_predictions,
	}

	start_time = time.time()

	loss = []
	perplexity_name = []
	perplexity_notName = []
	perplexity = []
	rank_predictions = []

	for step in range(lm_data.epoch_size):
		lm_x, lm_y = next(lm_data.generator)
		switch_y = next(switch_data)
		feed_dict = {
			model.lm_x: lm_x,
			model.lm_y: lm_y,
			model.switch_y: switch_y,
		}

		for i, (c, h) in enumerate(model.initial_states):
			feed_dict[c] = states[i].c
			feed_dict[h] = states[i].h

		results = session.run(fetches, feed_dict)
		states = results["final_states"]
		loss.append( results["loss"] )
		perplexity_name.append( results["perplexity_name"] )
		perplexity_notName.append( results["perplexity_notName"] )
		perplexity.append( results["perplexity"] )
		rank_predictions.extend( results["rank_predictions"] )

		with open(outFile_name, "a") as fout:
			name_indices = np.where(switch_y==1)[0]
			for ind in name_indices:
				row = ind//num_steps
				col = ind - row*num_steps
				words = [id2word[w] for w in lm_y[row, col-5:col+1]]
				fout.write("\n"+str(" ".join(words))+"\n")
				predictions = [id2word[p] for p in results["top_predictions"][ind,:]]
				fout.write("- Top 10 predictions: "+str(" ".join(predictions))+"\n")
				fout.write("- Rank true answer: "+str(results["rank_predictions"][ind])+"\n")
				fout.write("- Perplexity: "+str(np.exp(results["perplexities"][ind]))+"\n")

	loss = np.mean(loss)
	perplexity_name = np.exp(np.mean(perplexity_name))
	perplexity_notName = np.exp(np.mean(perplexity_notName))
	perplexity = np.exp(np.mean(perplexity))
	rank_predictions = np.median(rank_predictions)

	with open(outFile_name, "a") as fout:
		fout.write("\n"+"- Evaluation (corpus-wise, averaged) results:"+"\n")
		fout.write(" * Total loss = "+str(loss)+"\n")
		fout.write(" * Perplexity = "+str(perplexity)+" , Perplexity name = "+str(perplexity_name)+" , Perplexity notName = "+str(perplexity_notName)+"\n")
		fout.write(" * Prediction rank (median)  = "+str(rank_predictions))


def get_word2vec(train_path, vector_dim, word2id):
	# Load if already exists
	pickle_path = os.path.split(train_path)[0]+"/preprocessed/word2vec_dim"+str(vector_dim)+"_vocSize"+str(len(word2id))+".pkl"
	if os.path.exists(pickle_path):
		print("Restoring word2vec from: "+pickle_path+" \n")
		with open(pickle_path, "rb") as f:
			return pickle.load(f)

	# Train sentences generator
	class SentenceGenerator(object):
		def __init__(self, train_path):
			self.train_path = train_path
		
		def __iter__(self):
			for line in open(self.train_path, "r", encoding="utf-8"):
				yield tokenizer(line)
				
	sentence_generator = SentenceGenerator(train_path)
	print("Training word2vec...")
	model = gensim.models.Word2Vec(sentence_generator, size=vector_dim)
	print("Done :)")

	vectors = np.empty([len(word2id), vector_dim], dtype="float32")
	for word in word2id.keys():
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


def write_summary(summary_writer, global_step, values):
	list_values = []
	for key, value in values.items():
		list_values.append(tf.Summary.Value(tag=key, simple_value=value)) # TODO: Support other types of values (e.g. histogram)

	new_summ = tf.Summary()
	new_summ.value.extend(list_values)
	summary_writer.add_summary(new_summ, global_step)