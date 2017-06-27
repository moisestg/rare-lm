import nltk
from nltk.tag import StanfordPOSTagger
import numpy as np
import matplotlib.pyplot as plt
#from pycorenlp import StanfordCoreNLP


## FUNC'S DEFINITION ##

def tokenizer(line):
	return line.split()

def target_in_context(data):
	lemma = nltk.wordnet.WordNetLemmatizer()
	mention = np.array([])
	distance = np.array([])
	repetition = np.array([])
	for line in data:
		words = tokenizer(line)
		words = [ lemma.lemmatize(word) for word in words ]
		context = np.array(words[:-1])
		target = words[-1]
		mention = np.append(mention, "Y" if target in context else "N")
		mention_indexes = np.where(context == target)[0]
		if len(mention_indexes) == 0: 
			distance = np.append(distance, -1)
			repetition = np.append(repetition, -1)
		else:
			distance = np.append(distance, np.mean(len(words)-mention_indexes))
			repetition = np.append(repetition, len(mention_indexes))
	return [mention, distance, repetition]

def pos_tags(data, lib_path, caseless=False):
	if caseless:
		tagger = StanfordPOSTagger(lib_path+"models/english-caseless-left3words-distsim.tagger", path_to_jar=lib_path+"stanford-postagger-3.7.0.jar")
	else:
		tagger = StanfordPOSTagger(lib_path+"models/english-bidirectional-distsim.tagger", path_to_jar=lib_path+"stanford-postagger-3.7.0.jar")
	result = np.array([])
	for i, line in enumerate(data):
		print("Processing line "+str(i))
		words = tokenizer(line)
		tags = tagger.tag(words)
		result = np.append(result, tags[-1][1])
	return result


# def coref(data):
# 	# START THE SERVER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# 	stanford = StanfordCoreNLP("http://localhost:9000")
# 	for line in data:
# 		# Determine "position" last word
# 		splitted = stanford.annotate(line, properties={"annotators": "ssplit", "outputFormat": "json"})
# 		sentence_num = splitted["sentences"][-1]["index"] + 1 
# 		word_index = splitted["sentences"][-1]["tokens"][-1]["index"]
		
# 		# Find out if the target_word is involved in a coreference chain
# 		output = stanford.annotate(line, properties={
# 			"annotators": "tokenize,ssplit,pos,lemma,ner,parse,mention,coref",
# 			"coref.algorithm": "neural",
# 			"outputFormat": "json"
# 		})["corefs"]

# 		for chain_id in output:
# 			if any(ref["sentNum"] == sentence_num and ref["startIndex"] == word_index for ref in output[chain_id]):
# 				print(output[chain_id])
		

def rename_pos(tag):
	if tag=="NNP" or tag=="NNPS":
		return "PN"
	elif tag=="NN" or tag=="NNS":
		return "CN"
	elif tag=="JJ" or tag=="JJS" or tag=="JJR":
		return "ADJ"
	elif tag=="VB" or tag=="VBD" or tag=="VBG" or tag=="VBN" or tag=="VBP" or tag=="VBZ":
		return "V"
	elif tag=="RB" or tag=="RP" or tag=="RBS" or tag=="RBR":
		return "ADV"
	else:
		return "O"


# PLOTS

def autolabel(rects, ax):
	"""
	Attach a text label above each bar displaying its height
	"""
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2, 1.005*height, '%.2f' % height, ha='center', va='bottom')

def split_categories_plot(perp, acc, labels):
	# Group results
	perp_dict = {}
	acc_dict = {}
	for label in set(labels):
		perp_dict[label] = np.exp(np.mean( [val for i, val in enumerate(perp) if labels[i]==label] ))
		acc_dict[label] = 100*np.mean( [val for i, val in enumerate(acc) if labels[i]==label] )	
	perp_indexes = np.argsort(list(perp_dict.values()))[::-1]
	perp_vals = np.array(list(perp_dict.values()))[perp_indexes]
	perp_labels = np.array(list(perp_dict.keys()))[perp_indexes]
	acc_indexes = np.argsort(list(acc_dict.values()))[::-1]
	acc_vals = np.array(list(acc_dict.values()))[acc_indexes]
	acc_labels = np.array(list(acc_dict.keys()))[acc_indexes]
	# Generate plots
	ind = np.arange(len(perp_vals))  # the x locations for the groups
	width = 0.35 
	fig, ax = plt.subplots()
	rects = ax.bar(ind, perp_vals, width, color='b')	
	# add some text for labels, title and axes ticks
	ax.set_ylabel('Perplexity')
	ax.set_title('Perplexity')
	ax.set_xticks(ind) #  + width / 2
	ax.set_xticklabels(perp_labels)
	autolabel(rects, ax)
	plt.show()
	# other plot
	fig, ax = plt.subplots()
	rects = ax.bar(ind, acc_vals, width, color='r')
	ax.set_ylabel('Accuracy (%)')
	ax.set_title('Accuracy')
	ax.set_xticks(ind) #  + width / 2
	ax.set_xticklabels(acc_labels)
	autolabel(rects, ax)
	plt.show()

def bins_distance(distance):
	distance_bins = np.array([])	
	for i, num in enumerate(distance):
		if num == -1:
			distance_bins = np.append(distance_bins, "N") 
		elif 1 <= num and num <= 10:
			distance_bins = np.append(distance_bins, "[1,10]")
		elif 10 < num and num <= 20:
			distance_bins = np.append(distance_bins, "(10,20]")
		elif 20 < num and num <= 30:
			distance_bins = np.append(distance_bins, "(20,30]")
		elif 30 < num and num <= 40:
			distance_bins = np.append(distance_bins, "(30,40]")
		elif 40 < num and num <= 50:
			distance_bins = np.append(distance_bins, "(40,50]")
		elif 50 < num and num <= 60:
			distance_bins = np.append(distance_bins, "(50,60]")
		elif 60 < num and num <= 70:
			distance_bins = np.append(distance_bins, "(60,70]")
		elif 70 < num and num <= 80:
			distance_bins = np.append(distance_bins, "(70,80]")
		elif 80 < num:
			distance_bins = np.append(distance_bins, "+80")
	return distance_bins

## MAIN ##

if __name__ == "__main__":

	# Load data
	test_path = "./analysis/lambada_test_data_capitalized_plain_text.txt"
	with open(test_path, "r", encoding="utf-8") as f:
		test_data = f.readlines()
	test_data = [*map(str.strip, test_data)]

	dev_path = "./analysis/lambada_dev_data_capitalized_plain_text.txt"
	with open(dev_path, "r", encoding="utf-8") as f:
		dev_data = f.readlines()
	dev_data = [*map(str.strip, dev_data)]

	control_path = "./analysis/lambada_control_test_data_plain_text.txt"
	with open(control_path, "r", encoding="utf-8") as f:
		control_data = f.readlines()
	control_data = [*map(str.strip, control_data)]

	lambada = test_data + dev_data

	## TEST SET ##

	# Target word in context or not
	# Yes: 84.67%, No: 15.33% (out of a total of 5153 examples)
	test_context, test_distance, test_repetition = target_in_context( [*map(str.lower, test_data)] )
	fd = nltk.FreqDist(test_context)
	fd.tabulate()

	# Distance to mention
	# 41-50 31-40  N  51-60 21-30 11-20 61-70 71-80  +80   1-10 
    #  863   797  790  728   678   504   461   189    69    13 
	plt.hist(test_distance, bins=50)
	plt.show()

	test_distance = bins_distance(test_distance)
	fd = nltk.FreqDist(test_distance)
	fd.tabulate()

	# Number of mentions
	#   1    2    N    3     4    5 
	# 3367  837  790  135   23    1
	test_repetition = np.array( ["N" if num == -1 else int(num) for num in test_repetition] )
	fd = nltk.FreqDist(test_repetition)
	fd.tabulate()

	np.save("./analysis/test_context", test_context)
	np.save("./analysis/test_distance", test_distance)
	np.save("./analysis/test_repetition", test_repetition)

	# Target word PoS tag (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
	# Capitalized text: PN: 43.62%, CN: 42.98%, V: 7.12%, ADJ: 4.31%, ADV: 1.28%, O: 0.68% (out of a total of 5153 examples)
	# Caseless text: PN: 44.07%, CN: 40.99%, V: 7.47%, ADJ: 5.12%, ADV: 1.36%, O: 0.9% (out of a total of 5153 examples)
	test_pos = pos_tags(test_data, "/home/moises/thesis/stanford-postagger-full-2016-10-31/", caseless=False)
	test_pos = np.array([*map(rename_pos, test_pos)])
	fd = nltk.FreqDist(test_pos)
	fd.tabulate()

	np.save("./analysis/test_pos", test_pos)

	## CONTROL SET ##
	
	control_context, control_distance, control_repetition = target_in_context( [*map(str.lower, control_data)] )
	
	# Target word in context or not
	#   N    Y 
	# 4163  837 
	fd = nltk.FreqDist(control_context)
	fd.tabulate()

	# Distance to mention
	#   N (30,40] (20,30] (40,50] (10,20] (50,60] (60,70]  [1,10] (70,80]     +80 
	# 4163     155     141     137     110      97      83      47      43      24
	plt.hist(control_distance, bins=50)
	plt.show()
	control_distance = bins_distance(control_distance)
	fd = nltk.FreqDist(control_distance)
	fd.tabulate()

	# Number of mentions
	#   N    1    2    3    4    5    6    7    9    8 
	# 4163  572  163   57   27   10    3    3    1    1 
	control_repetition = np.array( ["N" if num == -1 else int(num) for num in control_repetition] )
	fd = nltk.FreqDist(control_repetition)
	fd.tabulate()

	# Target word PoS tag

	control_pos = pos_tags(control_data, "/home/moises/thesis/stanford-postagger-full-2016-10-31/", caseless=True)
	control_pos = np.array([*map(rename_pos, control_pos)])
	fd = nltk.FreqDist(control_pos)
	fd.tabulate()

	np.save("./analysis/control_context", control_context)
	np.save("./analysis/control_distance", control_distance)
	np.save("./analysis/control_repetition", control_repetition)
	np.save("./analysis/control_pos", control_pos)