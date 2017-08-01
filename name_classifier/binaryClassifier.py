import tensorflow as tf

class BinaryClassifier(object):
	"""
	A simple binary classifier to predict name/no name.
	Uses a fully connected layer with ReLU and a softmax layer.
	"""
	def __init__(self, is_training, config):
		hidden_size = config.hidden_size
		# Minibatch placeholders for input and output
		# The output from the LSTM
		self.input_x = input_x = tf.placeholder(tf.int32, [None, hidden_size], name="input_x")
		# The target (name or not name)
		self.input_y = input_y = tf.placeholder(tf.int64, [None], name="input_y") 
					
		# Fully connected layer with ReLU 
		with tf.name_scope("relu_layer"):
			d_prime = hidden_size
			W_h = tf.get_variable("W_h", [hidden_size, d_prime], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			b_h = tf.get_variable("b_h", [d_prime], tf.float32, initializer=tf.zeros_initializer())
			x_h = tf.nn.relu(tf.matmul(self.input_x, W_h) + b_h)
			
		# Soft-max layer
		with tf.name_scope("softmax"):
			W = tf.get_variable("W", [d_prime, 2], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable("b", [2], tf.float32, initializer=tf.zeros_initializer())
			self.logits = tf.matmul(x_h, W) + b # [batch, num_classes]
			self.predictions = tf.argmax(self.logits, 1, name="predictions")
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
			self.loss = tf.reduce_mean(losses)

		# Calculate batch stats
		#with tf.name_scope("batch_stats"):
		#	correct_predictions = tf.equal(self.predictions, self.input_y)
		#	self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
		
		if not is_training:
			return

		with tf.name_scope("optimizer"):
			optimizer = tf.train.AdamOptimizer(config.learning_rate)
			grads_and_vars = optimizer.compute_gradients(self.loss)
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.contrib.framework.get_or_create_global_step())
		
