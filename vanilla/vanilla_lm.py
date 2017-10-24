import tensorflow as tf
import lambada_utils as utils

"""

Multilayer LSTM language model

"""
class VanillaLM(object):

	def __init__(self, config, is_training, pretrained_emb=None):
		# Config parameters
		if is_training:
			self.num_steps = num_steps = config.num_steps_train
			self.batch_size = batch_size = config.batch_size_train
		else:
			self.num_steps = num_steps = config.num_steps_eval
			self.batch_size = batch_size = config.batch_size_eval
		self.emb_keep_prob = emb_keep_prob = config.emb_keep_prob
		self.input_keep_prob = input_keep_prob = config.input_keep_prob
		self.output_keep_prob = output_keep_prob = config.output_keep_prob
		self.state_keep_prob = state_keep_prob = config.state_keep_prob
		self.l2_reg = l2_reg = config.l2_reg
		self.num_layers = num_layers = config.num_layers
		self.hidden_size = hidden_size = config.hidden_size
		self.projection_size = projection_size = config.projection_size
		if projection_size:
			final_hidden_size = projection_size
		else:
			final_hidden_size = hidden_size
		self.emb_size = emb_size = config.emb_size
		self.vocab_size = vocab_size = config.vocab_size
		self.clip_norm = clip_norm = config.clip_norm
		self.optimizer_algo = optimizer_algo = config.optimizer_algo

		# Placeholders
		self.lm_x = lm_x = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps], name="lm_x")
		self.lm_y = lm_y = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps], name="lm_y")
		self.switch_y = switch_y = tf.placeholder(dtype=tf.float32, shape=[batch_size*num_steps], name="switch_y") # float32 cause sigmoid_loss
		self.lr = tf.Variable(config.learning_rate, trainable=False)
		
		# Embedding layer
		with tf.name_scope("embedding_layer"):
			with tf.device("/cpu:0"):
				if pretrained_emb is not None:
					embedding = tf.get_variable("embedding", dtype=tf.float32, initializer=pretrained_emb)
				else:
					embedding = tf.get_variable("embedding", dtype=tf.float32, shape=[vocab_size, emb_size])
				if is_training and emb_keep_prob < 1:
					final_embedding = tf.nn.dropout(embedding, keep_prob=emb_keep_prob, noise_shape=[vocab_size, 1]) # embedding dropout
				else:
					final_embedding = embedding
				lm_x = tf.nn.embedding_lookup(final_embedding, lm_x) # [batch_size, num_steps, emb_size]
			lm_x = tf.unstack(lm_x, num=num_steps, axis=1) # list of num_steps Tensors of shape [batch_size, emb_size]

		# Recurrent layer(s)
		with tf.name_scope("recurrent_layer"):
			def lstm_cell():
				cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
				if is_training and (input_keep_prob < 1 or output_keep_prob < 1 or state_keep_prob < 1):
					cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob, input_keep_prob=input_keep_prob,
					input_size=emb_size, state_keep_prob=state_keep_prob,
					variational_recurrent=True, dtype=tf.float32) # Variational Dropout
				if projection_size:
					cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=projection_size)
				return cell

			cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)

			self.initial_states = cell.zero_state(batch_size, tf.float32)
			hidden_states, states = tf.contrib.rnn.static_rnn(cell, lm_x, initial_state=self.initial_states) # hidden_states: list of num_steps tensors of shape [batch_size, final_hidden_size ]
			self.hidden_states = hidden_states
			self.final_states = states

			hidden_states_flattened = tf.reshape(tf.concat(values=hidden_states, axis=1), [-1, final_hidden_size])

		# Softmax layer
		lm_y_flattened = tf.reshape(lm_y, [-1])
		target_indices = tf.stack([tf.range(batch_size*num_steps), lm_y_flattened], axis=1)

		with tf.name_scope("softmax_layer"): 
			W_softmax = tf.get_variable("W_softmax", dtype=tf.float32, shape=[final_hidden_size, vocab_size]) #, initializer=tf.contrib.layers.xavier_initializer()
			b_softmax = tf.get_variable("b_softmax", dtype=tf.float32, shape=[vocab_size], initializer=tf.zeros_initializer())
			logits_softmax = tf.matmul(hidden_states_flattened, W_softmax) + b_softmax
			prob_softmax_all = utils.softmax_stable(logits_softmax)
			self.prob_softmax = prob_softmax = utils.softmax_stable_target(logits_softmax, target_indices) # [batch_size*num_steps]

			indices_name = tf.where(tf.equal(switch_y, 1))
			indices_notName = tf.where(tf.equal(switch_y, 0))

		# Loss calculation
		self.perplexity_name =  -1. * tf.reduce_mean(utils.log_stable( tf.gather(prob_softmax, indices_name)))
		self.perplexity_notName = -1. * tf.reduce_mean(utils.log_stable( tf.gather(prob_softmax, indices_notName)))
		
		self.perplexities = perplexities = -1. * utils.log_stable(prob_softmax) # actually log-perplexities
		self.lm_loss = lm_loss =  tf.reduce_sum(perplexities) / batch_size # average-batch cross entropy loss
		self.perplexity = lm_loss / num_steps # still has to be exponentiated to get perplexity

		self.reg_loss = reg_loss = tf.nn.l2_loss(W_softmax)

		self.loss = loss = lm_loss + l2_reg*reg_loss
		
		# Additional stats (rank and top predictions)
		#self.jsd = utils.jsd(prob_softmax_name, prob_softmax_notName)
		self.jsd = tf.constant(0.5) # TODO: Change this

		sorted_indices = tf.nn.top_k(prob_softmax_all, vocab_size).indices
		self.top_predictions = sorted_indices[:,:10] # [batch_size*num_steps, 10]
		self.rank_predictions = rank_predictions = tf.where(tf.equal(sorted_indices, tf.expand_dims(lm_y_flattened,1)))[:,1]
		#self.rank_prediction = rank_prediction = tf.reduce_mean(rank_predictions) # median?
		#self.rank_prediction_name = tf.reduce_mean(tf.gather(rank_predictions, indices_name))
		#self.rank_prediction_notName = tf.reduce_mean(tf.gather(rank_predictions, indices_notName))

		if not is_training:
			return

		# Training operation
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clip_norm)
		if optimizer_algo == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		elif optimizer_algo == "sgd":
			optimizer = tf.train.GradientDescentOptimizer(self.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

		# Possibility to update the learning rate
		self.new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_lr")
		self.lr_update = tf.assign(self.lr, self.new_lr)

	# Function to change the learning rate of the optimizer
	def assign_lr(self, session, new_lr):
		session.run(self.lr_update, feed_dict={self.new_lr: new_lr})