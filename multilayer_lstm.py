import tensorflow as tf
import collections

class MultilayerLSTM(object):
	"""Multilayer LSTM baseline"""

	def __init__(self, is_training, config, pretrained_emb=None):

		batch_size = config.batch_size
		num_steps = config.num_steps
		self.input_x = input_x = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_x")
		self.input_y = input_y = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_y")

		## EMBEDDING ##
		with tf.device("/cpu:0"):
			if pretrained_emb is not None:
				embedding = tf.get_variable("embedding", dtype=tf.float32, initializer=pretrained_emb)
			else:
				embedding = tf.get_variable("embedding", [config.vocab_size, config.emb_size], dtype=tf.float32)

			input_x = tf.nn.embedding_lookup(embedding, input_x)

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
		self.final_state = state

		## SOFTMAX ##
		outputs_flat = tf.reshape(tf.concat(axis=1, values=outputs), [-1, config.hidden_size]) # [batch_size*num_steps, hidden_size]
		softmax_w = tf.get_variable("softmax_w", [config.hidden_size, config.vocab_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		softmax_b = tf.get_variable("softmax_b", [config.vocab_size], dtype=tf.float32)
		self.logits = logits = tf.matmul(outputs_flat, softmax_w) + softmax_b # [batch_size*num_steps, vocab_size]
		#self.probs = tf.nn.softmax(logits)
		#outputs_flattened = tf.reshape(self._outputs, [-1]) # [batch_size*num_steps]
		#indexes = tf.range(0, batch_size*num_steps)
		#select_indexes = tf.stack([indexes, outputs_flattened], axis=1) # [batch_size*num_steps, 2] 
		#relevant_probs = tf.gather_nd(self.probs, select_indexes) # [batch_size*num_steps, vocab_size]

		# Populate cache
		#inputs_flattened = tf.reshape(self._inputs, [-1]).eval(tf.get_default_session()) 
		
		self.loss = loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(self.input_y, [-1])],
				[tf.ones([batch_size * num_steps], dtype=tf.float32)])

		#self._loss = loss = -tf.log(relevant_probs)
		self.cost = cost = tf.reduce_sum(loss) / batch_size
		

		## STATS ##
		predictions = tf.to_int32(tf.argmax(logits, 1, name="predictions")) # tf.argmax returns int64
		correct_predictions = tf.equal(predictions, tf.reshape(self.input_y, [-1]))
		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		if not is_training:
			return


		## OPTIMIZER ##

		self.lr = tf.Variable(config.learning_rate, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.clip_norm) 
		
		if config.optimizer == "grad_desc":
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
		elif config.optimizer == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		else:
			raise ValueError("Not supported optimizer :(")

		self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

		self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self.lr_update = tf.assign(self.lr, self.new_lr)

	# Function to change the learning rate of the optimizer
	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})