import tensorflow as tf
import collections

class MultilayerLSTM(object):
	"""Multilayer LSTM baseline"""

	def __init__(self, is_training, config, pretrained_emb=None):

		num_steps = config.num_steps
		self.input_x = input_x = tf.placeholder(tf.int32, [None, num_steps], name="input_x")
		self.input_y = input_y = tf.placeholder(tf.int32, [None, num_steps], name="input_y")
		self.name_y = name_y = tf.placeholder(tf.int64, [None], name="name_y") # (name 1 or not name 0)
		self.batch_size = batch_size = tf.placeholder(tf.int32, name="batch_size")

		## EMBEDDING ##

		with tf.device("/cpu:0"):
			if pretrained_emb is not None:
				embedding = tf.get_variable("embedding", dtype=tf.float32, initializer=pretrained_emb)
			else:
				embedding = tf.get_variable("embedding", [config.vocab_size, config.emb_size], dtype=tf.float32)

			input_x = tf.nn.embedding_lookup(embedding, input_x) # [batch_size, num_steps, emb_size]

		if is_training and config.keep_prob < 1:
			input_x = tf.nn.dropout(input_x, config.keep_prob)

		input_x = tf.unstack(input_x, num=num_steps, axis=1) # list of num_steps Tensors of shape [batch_size, emb_size]


		## RNN LAYER ##

		def lstm_cell():
			cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

			if is_training and config.keep_prob < 1:
				return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob) # Regularization as in https://arxiv.org/pdf/1409.2329.pdf 
			else:
				return cell

		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(config.num_layers)], state_is_tuple=True)

		self.initial_state = cell.zero_state(batch_size, tf.float32)

		outputs, state = tf.contrib.rnn.static_rnn(cell, input_x, initial_state=self.initial_state) # outputs is a list of num_steps Tensors of shape [batch_size, hidden_size] / slower: tf.nn.dynamic_rnn()
		self.outputs = outputs
		self.outputs_stacked = tf.stack(outputs, axis=1)
		self.outputs_nameFormat = tf.reshape(self.outputs_stacked, [-1, config.hidden_size])
		self.final_state = state


		## SOFTMAX ##

		outputs_flat = tf.reshape(tf.concat(axis=1, values=outputs), [-1, config.hidden_size]) # [batch_size*num_steps, hidden_size]
		softmax_w = tf.get_variable("softmax_w", [config.hidden_size, config.vocab_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		softmax_b = tf.get_variable("softmax_b", [config.vocab_size], dtype=tf.float32)
		self.logits = logits = tf.matmul(outputs_flat, softmax_w) + softmax_b # [batch_size*num_steps, vocab_size]
		
		self.loss = loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example( # [batch_size*num_steps]
				[logits],
				[tf.reshape(self.input_y, [-1])],
				[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		
		self.cost = cost = tf.reduce_sum(loss) / tf.to_float(batch_size) # later in train.py also divided by num_steps (as expected)
		

		## STATS ##

		self.predictions = predictions = tf.to_int32(tf.argmax(logits, 1, name="predictions")) # tf.argmax returns int64
		self.correct_predictions = correct_predictions = tf.equal(predictions, tf.reshape(self.input_y, [-1]))
		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		## NAME PREDICTOR ## 
		# Fully connected layer with ReLU 
		with tf.name_scope("relu_layer"):
			d_prime = config.hidden_size # other values?
			W_h = tf.get_variable("W_h", [config.hidden_size, d_prime], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			b_h = tf.get_variable("b_h", [d_prime], tf.float32, initializer=tf.zeros_initializer())
			x_h = tf.nn.relu(tf.matmul(self.outputs_nameFormat, W_h) + b_h)
			
		# Soft-max layer
		with tf.name_scope("softmax"):
			W_name = tf.get_variable("W", [d_prime, 2], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			b_name = tf.get_variable("b", [2], tf.float32, initializer=tf.zeros_initializer())
			self.logits_name = tf.matmul(x_h, W_name) + b_name # [batch, num_classes]
			self.predictions_name = tf.argmax(self.logits_name, 1, name="predictions_name")
			losses_name = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_name, labels=self.name_y)
			self.loss_name = loss_name = tf.reduce_mean(losses_name)

		self.loss_sum = self.cost + self.loss_name

		if not is_training:
			return

		## OPTIMIZER ##

		self.lr = tf.Variable(config.learning_rate, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_sum, tvars), config.clip_norm) 
		
		if config.optimizer == "grad_desc":
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
		elif config.optimizer == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		elif config.optimizer == "adagrad":
			optimizer = tf.train.AdagradOptimizer(self.lr)
		else:
			raise ValueError("Not supported optimizer :(")

		self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

		self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self.lr_update = tf.assign(self.lr, self.new_lr)

	# Function to change the learning rate of the optimizer
	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})