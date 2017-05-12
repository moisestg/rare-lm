import os
from collections import Counter
import numpy as np
import pickle

_UNK = "<UNK>"
_EOS = "<EOS>"

"""
Very basic tokenizer
"""
def tokenizer(sentence):
	return sentence.strip().split(" ")

"""
Load (or build and save) vocabulary
"""
def load_vocabulary(train_path="", max_vocab_size=10000):
	cwd = os.getcwd()
	file_path = cwd+"/preprocessed/word2Id_"+str(max_vocab_size)+".pkl"
	if os.path.exists(file_path):
		with open(file_path, "rb") as f:
			return pickle.load(f)
	else:
		word_counts = Counter()
		for dir_path,_,files in os.walk(train_path):
			for file_name in files:
				with open(os.path.join(dir_path, file_name), "r") as sentences:
					for sentence in sentences:
						for word in tokenizer(sentence):
							if word in word_counts: 
								word_counts[word] += 1
							else:
								word_counts[word] = 1
	total_freq = sum(freq for freq in word_counts.values())
	kept_freq = sum(tupl[1] for tupl in word_counts.most_common(max_vocab_size-2)) # tupl -> (word, frequency)
	print("Vocabulary of size "+str(max_vocab_size)+" (real size: "+str(max_vocab_size-2)+") covers "+str(round(kept_freq/total_freq*100, 2))+"% of the training words.")
	vocabulary = [tupl[0] for tupl in word_counts.most_common(max_vocab_size-2)] # including the 2 special tags  
	word2Id = {word: i for i, word in enumerate(vocabulary, 2)}
	word2Id[_UNK] = 0
	word2Id[_EOS] = 1
	if not os.path.exists(cwd+"/preprocessed"):
		os.makedirs(cwd+"/preprocessed")
	with open(file_path, "wb") as f:
		pickle.dump(word2Id, f)
	return word2Id

"""
Loads data and returns the respective ids for words
"""
def load_data(dataset="train", num_steps=20, max_vocab_size=10000, data_path=None, word2Id=None):
	cwd = os.getcwd()
	file_name = cwd+"/preprocessed/x_"+dataset+"_vocSize"+str(max_vocab_size)+"_steps"+str(num_steps)+".pkl"
	#file_y = cwd+"/preprocessed/y_"+dataset+"_vocSize"+str(max_vocab_size)+"_steps"+str(num_steps)+".pkl"
	# If files exist, load them
	if os.path.exists(file_name):
		with open(file_name, "rb") as f:
			data = pickle.load(f)
		return data
	# If not, construct training data
	else: 
		data = []
		for dir_path,_,files in os.walk(data_path):
			for file_name in files:
				with open(os.path.join(dir_path, file_name), "r") as sentences:
					for sentence in sentences:
						for word in tokenizer(sentence.replace("\n", _EOS)):
							data.append(word2Id[word] if word in word2Id else word2Id[_UNK])
		data = np.array(data)
		if not os.path.exists(cwd+"/preprocessed"):
			os.makedirs(cwd+"/preprocessed")
		with open(file_name, "wb") as f:
			pickle.dump(data, f)
		return data


def batch_iterator(raw_data, batch_size, num_steps, num_epochs):
	data_len = len(raw_data)
	num_batches = data_len // batch_size
	data = np.reshape(raw_data[0 : batch_size * num_batches], [batch_size, num_batches])
	epoch_size = (num_batches - 1) // num_steps

	for i in list(range(0, epoch_size))*num_epochs:
		x_batch = data[:, i*num_steps:(i+1)*num_steps]
		y_batch = data[:, i*num_steps+1:(i+1)*num_steps+1]
		yield [x_batch, y_batch]


# def batch_producer(raw_data, batch_size, num_steps, name=None):
# 	"""Iterate on the raw PTB data.
# 	This chunks up raw_data into batches of examples and returns Tensors that
# 	are drawn from these batches.
# 	Args:
# 		raw_data: one of the raw data outputs from ptb_raw_data.
# 		batch_size: int, the batch size.
# 		num_steps: int, the number of unrolls.
# 		name: the name of this operation (optional).
# 	Returns:
# 		A pair of Tensors, each shaped [batch_size, num_steps]. The second element
# 		of the tuple is the same data time-shifted to the right by one.
# 	Raises:
# 		tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
# 	"""
# 	with tf.name_scope(name, "batchProducer", [raw_data, batch_size, num_steps]):
# 		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

# 		data_len = tf.size(raw_data)
# 		batch_len = data_len // batch_size
# 		data = tf.reshape(raw_data[0 : batch_size * batch_len],
# 											[batch_size, batch_len])

# 		epoch_size = (batch_len - 1) // num_steps
# 		assertion = tf.assert_positive(
# 				epoch_size,
# 				message="epoch_size == 0, decrease batch_size or num_steps")
# 		with tf.control_dependencies([assertion]):
# 			epoch_size = tf.identity(epoch_size, name="epoch_size")

# 		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
# 		x = tf.strided_slice(data, [0, i * num_steps],
# 												 [batch_size, (i + 1) * num_steps])
# 		x.set_shape([batch_size, num_steps])
# 		y = tf.strided_slice(data, [0, i * num_steps + 1],
# 												 [batch_size, (i + 1) * num_steps + 1])
# 		y.set_shape([batch_size, num_steps])
# 		return x, y

"""
Generates a batch iterator for a dataset
"""
# def batch_iter(data, batch_size, num_epochs, shuffle=True):
# 	data = np.array(data)
# 	data_size = len(data)
# 	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
# 	for epoch in range(num_epochs):
# 		# Shuffle the data at each epoch
# 		if shuffle:
# 			shuffle_indices = np.random.permutation(np.arange(data_size))
# 			shuffled_data = data[shuffle_indices]
# 		else:
# 			shuffled_data = data
# 		for batch_num in range(num_batches_per_epoch-1): # TODO: Fix last batch (remove -1)
# 			start_index = batch_num * batch_size
# 			end_index = min((batch_num + 1) * batch_size, data_size)
# 			yield shuffled_data[start_index:end_index]