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
	file_x = cwd+"/preprocessed/x_"+dataset+"_vocSize"+str(max_vocab_size)+"_steps"+str(num_steps)+".pkl"
	file_y = cwd+"/preprocessed/y_"+dataset+"_vocSize"+str(max_vocab_size)+"_steps"+str(num_steps)+".pkl"
	# If files exist, load them
	if os.path.exists(file_x) and os.path.exists(file_y):
		with open(file_x, "rb") as f:
			x = pickle.load(f)
		with open(file_y, "rb") as f:
			y = pickle.load(f)
		return [x, y]
	# If not, construct training data
	else: 
		x = []
		for dir_path,_,files in os.walk(data_path):
			for file_name in files:
				with open(os.path.join(dir_path, file_name), "r") as sentences:
					for sentence in sentences:
						for word in tokenizer(sentence.replace("\n", _EOS)):
							x.append(word2Id[word] if word in word2Id else word2Id[_UNK])
		# Generate training data points of lenght "num_steps"
		n = len(x) % num_steps
		if n == 0:
			x = x[:-(num_steps-1)]
		else:
			x = x[:(1-n)]
		y = np.array( [x[i:i + num_steps] for i in range(1, len(x), num_steps)] )
		x = np.array( [x[i:i + num_steps] for i in range(0, len(x)-1, num_steps)] )

		if not os.path.exists(cwd+"/preprocessed"):
			os.makedirs(cwd+"/preprocessed")
		with open(file_x, "wb") as f:
			pickle.dump(x, f)
		with open(file_y, "wb") as f:
			pickle.dump(y, f)
		return [x, y]


"""
Generates a batch iterator for a dataset
"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]