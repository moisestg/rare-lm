import tensorflow as tf
import lambada_utils as utils

"""

Implementation of "Pointer Sentinel Mixture Models" (https://arxiv.org/abs/1609.07843)

"""
class PointerSentinelMixtureLM(object):

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
		self.attention_length = attention_length = config.attention_length

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
			prob_softmax = utils.softmax_stable_target(logits_softmax, target_indices) # [batch_size*num_steps]

			indices_name = tf.where(tf.equal(switch_y, 1))
			indices_notName = tf.where(tf.equal(switch_y, 0))
			#self.prob_softmax_name = prob_softmax_name = tf.reduce_mean(tf.gather(prob_softmax_all, indices=indices_name, axis=0), axis=0)
			#self.prob_softmax_notName = prob_softmax_notName = tf.reduce_mean(tf.gather(prob_softmax_all, indices=indices_notName, axis=0), axis=0)

		# Pointer layer
		with tf.name_scope("pointer_layer"):
			initial_attention_states = self.initial_attention_states = tf.get_variable("initial_attention_states", dtype=tf.float32, shape=[batch_size, final_hidden_size, attention_length], trainable=False) # OJO: initializer=tf.zeros_initializer(),
			initial_attention_ids = self.initial_attention_ids = tf.get_variable("initial_attention_ids", dtype=tf.int32, shape=[batch_size, attention_length], initializer=tf.zeros_initializer(), trainable=False) # initial ids (0, PAD) are valid but ignored
			W_query = tf.get_variable("W_query", dtype=tf.float32, shape=[final_hidden_size, final_hidden_size])
			b_query = tf.get_variable("b_query", dtype=tf.float32, shape=[final_hidden_size])
			sentinel = tf.get_variable("sentinel", dtype=tf.float32, shape=[final_hidden_size], initializer=tf.random_uniform_initializer(-.01, .01)) #

			attention_cacheProbs_all = [] # list of length "num_steps" of Tensors with shape [batch_size, attention_length]
			attention_ids_all = [] # ? or just slice from lm_x
			gates_all = [] # 

			attention_scores_all = []

			for step in range(num_steps):
				current_hidden_states = hidden_states[step] # [batch_size, final_hidden_size]
				current_ids = self.lm_x[:, step] # [batch_size]
				initial_attention_states = tf.concat([initial_attention_states[:,:,1:], tf.expand_dims(current_hidden_states, axis=-1)], axis=-1) # Update attention_states with current_hidden_states
				initial_attention_ids = tf.concat([initial_attention_ids[:,1:], tf.expand_dims(current_ids, axis=-1)], axis=-1)
				attention_ids_all.append( initial_attention_ids )
				queries = tf.matmul(current_hidden_states, W_query) + b_query # [batch_size, final_hidden_size]
				attention_cacheScores = tf.reduce_sum( initial_attention_states * tf.expand_dims(queries, axis=-1) , axis=1) # [batch_size, attention_length]
				attention_sentinelScores = tf.reduce_sum( queries * sentinel , axis=-1) # [batch_size]
				attention_scores = tf.concat([attention_cacheScores, tf.expand_dims(attention_sentinelScores, axis=-1)], axis=-1) # [batch_size, attention_length + 1]
				attention_scores_all.append(attention_scores)
				attention_probs = utils.softmax_stable(attention_scores) # [batch_size, attention_length + 1] (softmax along last dimension)	
				gates = attention_probs[:,-1] # [batch_size]
				gates_all.append( gates )
				attention_cacheProbs = attention_probs[:,:-1] # [batch_size, attention_length]
				attention_cacheProbs_all.append( attention_cacheProbs )

			self.final_attention_states = initial_attention_states
			self.final_attention_ids = initial_attention_ids

		# Mix probability distributions
		self.attention_scores = tf.reshape(tf.stack(attention_scores_all, axis=1), shape=[-1, attention_length+1])
		self.gates_all = gates_all = tf.reshape(tf.stack(gates_all, axis=1), shape=[-1]) # [batch_size*num_steps]
		self.gate_names = tf.reduce_mean( tf.gather(gates_all, indices_name) )
		self.gate_notNames = tf.reduce_mean( tf.gather(gates_all, indices_notName) )
		self.attention_cacheProbs_all = attention_cacheProbs_all = tf.reshape(tf.stack(attention_cacheProbs_all, axis=1), shape=[-1, attention_length]) # [batch_size*num_steps, attention_length]
		self.attention_ids_all = attention_ids_all = tf.reshape(tf.stack(attention_ids_all, axis=1), shape=[-1, attention_length]) # [batch_size*num_steps, attention_length]
		lm_y_flattened_tiled = tf.tile(tf.expand_dims(lm_y_flattened, axis=-1), multiples=[1, attention_length])
		cacheProbs_relevantIndexes = tf.where(tf.equal(attention_ids_all, lm_y_flattened_tiled))
		self.prob_pointer = prob_pointer = tf.reduce_sum( 
				tf.scatter_nd(indices=cacheProbs_relevantIndexes, 
				updates=tf.gather_nd(attention_cacheProbs_all, cacheProbs_relevantIndexes), 
				shape=attention_cacheProbs_all.get_shape()) 
			, axis=-1)

		self.prob_mix = prob_mix = gates_all * prob_softmax + prob_pointer

		# Loss calculation
		self.perplexity_name =  -1. * tf.reduce_mean(utils.log_stable( tf.gather(prob_mix, indices_name)))
		self.perplexity_notName = -1. * tf.reduce_mean(utils.log_stable( tf.gather(prob_mix, indices_notName)))
		
		self.perplexities = perplexities = -1. * utils.log_stable(prob_mix) # actually log-perplexities
		self.lm_loss = lm_loss =  tf.reduce_sum(perplexities) / batch_size # average-batch cross entropy loss
		self.perplexity = lm_loss / num_steps # still has to be exponentiated to get perplexity

		self.pointer_loss = pointer_loss = -1. * tf.reduce_sum(utils.log_stable(gates_all + prob_pointer)) / batch_size 

		self.reg_loss = reg_loss = tf.nn.l2_loss(W_softmax)

		self.loss = loss = lm_loss + pointer_loss + l2_reg*reg_loss
		
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