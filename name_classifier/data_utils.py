import os
import pickle
import numpy as np
import itertools
import sklearn.metrics as skmetrics
import tensorflow as tf

# Batch generation
def getIdFromFile(string):
	return int(re.findall(r'\d+', string)[0])

class BatchGenerator:

	def __init__(self, data_set):
		self.data_set = data_set
		self.epoch_size = len(os.listdir("inputs/"+str(self.data_set)))
		self.generator = self.generator()

	def generator(self):
		for file in itertools.cycle(sorted(os.listdir("inputs/"+str(self.data_set)), key=getIdFromFile)):
			with open("inputs/"+str(self.data_set)+"/"+file, "rb") as f:
				x_batch = pickle.load(f)
			with open("outputs/"+str(self.data_set)+"/"+file, "rb") as f:
				y_batch = pickle.load(f)
			yield x_batch, y_batch

	def getBatch(self):
		return next(self.generator)

# Summaries
def write_summary(summary_writer, current_step, values):
	list_values = []
	for key, value in values.items():
		list_values.append(tf.Summary.Value(tag=key, simple_value=value)) # TODO: Support other types of values (e.g. histogram)

	new_summ = tf.Summary()
	new_summ.value.extend(list_values)
	summary_writer.add_summary(new_summ, current_step)

# Evaluation
def getStats(y_true, y_pred):
	fscoreName, fscoreNoName = skmetrics.f1_score(y_true, y_pred, labels=[1,0], average=None)
	auc = skmetrics.roc_auc_score(y_true, y_pred, average=None)
	return fscoreName, fscoreNoName, auc

def eval_epoch(session, model, input_data, summary_writer=None):

	y_pred = []
	y_true = []

	fetches = {
			"predictions": model.predictions,
	}

	for step in range(input_data.epoch_size):
		input_x, input_y = input_data.getBatch()
		feed_dict = {
			model.input_x: input_x,
			model.input_y: input_y,
		}

		results = session.run(fetches, feed_dict)
		predictions = results["predictions"]

		y_pred.extend(predictions)
		y_true.extend(input_y)
	
	fscoreName, fscoreNoName, auc = getStats(y_true, y_pred)

	print("VALIDATION: fscore Name: %.3f, fscore NoName: %.3f, auc: %.3f" %
								(fscoreName, fscoreNoName, auc))

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), 
			{"fscoreName":fscoreName, "fscoreNoName":fscoreNoName, "auc":auc}) # Write summary (CORPUS-WISE stats)	

def debug_input(data_set):
	batch_size=64
	num_steps=35
	
	with open("../lambada-dataset/capitalized/"+data_set+".txt", "r") as f:
		data = f.readlines()

	lines = [line.strip().split() for line in data]
	words = [word for line in lines for word in line]

	# Reshape output
	data_len = len(words)
	batch_len = data_len // batch_size
	words = np.array(words)
	words = np.reshape(output[0 : batch_size * batch_len], [batch_size, batch_len])
	epoch_size = (batch_len - 1) // num_steps

	for i in itertools.cycle(range(epoch_size)):	
		y_batch = words[:, i*num_steps+1:(i+1)*num_steps+1].reshape([-1])
		yield y_batch

