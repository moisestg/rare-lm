import os
import re 
import pickle
import numpy as np

data_set="capitalized_train"
output_dir="outputs/"+data_set+"/"
batch_size=64
num_steps=35

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Load list of "verified" names
with open("final_names.pkl", "rb") as f:
	names_list = pickle.load(f)

# Load words
with open("../lambada-dataset/capitalized/"+data_set+".txt", "r") as f:
	data = f.readlines()

lines = [line.strip().split() for line in data]
words = [word for line in lines for word in line]

# Collect outputs
output = [1 if word in names_list else 0 for word in words]

# Reshape output
data_len = len(output)
batch_len = data_len // batch_size
output = np.array(output, dtype=np.int32)
output = np.reshape(output[0 : batch_size * batch_len], [batch_size, batch_len])
epoch_size = (batch_len - 1) // num_steps

for i in range(epoch_size):	
	y_batch = output[:, i*num_steps+1:(i+1)*num_steps+1].reshape([-1])
	with open(output_dir+data_set+"_outputs_batch"+str(i)+".pkl", "wb") as f:
		pickle.dump(y_batch, f)

