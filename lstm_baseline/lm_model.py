import tensorflow as tf
#import numpy as np

class BasicLSTM(object):
	"""
	Vanilla LSTM LM Baseline
	"""

	def __init__(self, vocab_size, embedding_size, num_steps, state_size, num_layers): # sequence_length, filter_sizes, num_filters, l2_reg_lambda=0.0
		
		# Placeholders for inputs and dropout probability
		self.input_x = tf.placeholder(tf.int32, [None, num_steps], name="input_x") # tf.int64?? 
		self.input_y = tf.placeholder(tf.int64, [None, num_steps], name="input_y") # list containing id of the output word
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.batch_size = tf.shape(self.input_x)[0]

		# Cell definition
		self.cell = tf.contrib.rnn.BasicLSTMCell(state_size) # , initializer=tf.contrib.layers.xavier_initializer() in tf.nn.rnn_cell in previous versions !
		self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.dropout_keep_prob)
		self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			self.W_emb = tf.get_variable("W_emb", [vocab_size, embedding_size], tf.float32, initializer=tf.random_uniform_initializer(-.1, .1))
			self.embedded_words = tf.nn.embedding_lookup(self.W_emb, self.input_x) # [None, num_steps, embedding_size]
			self.inputs = tf.nn.dropout(self.embedded_words, self.dropout_keep_prob) # IS THIS REALLY GOOD? Tensorflow tutorial

		# LSTM layer
		with tf.name_scope("rnn"):
			# TODO: Keep track of hidden state?? 
			self.init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
			outputs, last_state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.inputs, initial_state=self.init_state, dtype=tf.float32) # outputs: [batch_size, num_steps, state_size]
			self.outputs = tf.reshape(outputs, [-1, state_size]) # 3D to 2D: [batch_size*num_steps, state_size]
			#self.final_state = state

		# Soft-max layer
		with tf.name_scope("softmax"):
			W_sm = tf.get_variable("W_sm", [state_size, vocab_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			b_sm = tf.get_variable("b_sm", [vocab_size], tf.float32, initializer=tf.zeros_initializer())
			self.logits = tf.matmul(self.outputs, W_sm) + b_sm # [batch_size*num_steps, vocab_size]
			self.predictions = tf.argmax(self.logits, 1, name="predictions") # int64 tensor
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.input_y, [-1]))
			self.loss = tf.reduce_mean(losses)

		# Calculate accuracy
		with tf.name_scope("stats"):
			correct_predictions = tf.equal(self.predictions, tf.reshape(self.input_y, [-1]))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
			self.perplexity = tf.exp(self.loss, name="perplexity")
			