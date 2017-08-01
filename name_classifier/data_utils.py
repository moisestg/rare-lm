import os
import pickle
import numpy as np
import itertools
import sklearn.metrics as skmetrics

# Batch generation
class BatchGenerator:

	def __init__(self, data_set):
		self.data_set = data_set
		self.epoch_size = len(os.listdir("inputs/"+str(self.data_set)))
		self.generator = self.generator()

	def generator(self):
		for i in itertools.cycle(range(len(os.listdir("inputs/"+str(self.data_set))))):
			with open("inputs/"+str(self.data_set)+"/batch_"+str(i)+".pkl", "rb") as f:
				x_batch = pickle.load(f)
			with open("outputs/"+str(self.data_set)+"/batch_"+str(i)+".pkl", "rb") as f:
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
	aucName, aucNoName = skmetrics.roc_auc_score(y_true, y_pred, labels=[1,0], average=None)
	return fscoreName, fscoreNoName, aucName, aucNoName

def eval_epoch(session, model, input_data, summary_writer=None):

	y_pred = []
	y_true = []

	fetches = {
			"predictions": predictions,
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
	
	fscoreName, fscoreNoName, aucName, aucNoName = getStats(y_true, y_pred)

	print("VALIDATION: fscoreName: %.3f, fscoreNoName: %.3f, aucName: %.3f, aucNoName: %.3f" %
								(fscoreName, fscoreNoName, aucName, aucNoName))

	if summary_writer is not None:
		write_summary(summary_writer, tf.contrib.framework.get_or_create_global_step().eval(session), 
			{"fscoreName":fscoreName, "fscoreNoName":fscoreNoName, "aucName":aucName, "aucNoName":aucNoName}) # Write summary (CORPUS-WISE stats)	



