import nltk
from nltk.tag import StanfordPOSTagger
import numpy as np
import matplotlib.pyplot as plt


## FUNC'S DEFINITION ##

def tokenizer(line):
	return line.split()

def target_in_context(data):
	lemma = nltk.wordnet.WordNetLemmatizer()
	result = np.array([])
	for line in data:
		words = tokenizer(line)
		words = [ lemma.lemmatize(word) for word in words ]
		context = words[:-1]
		target = words[-1]
		result = np.append(result, "Y" if target in context else "N")
	return result

def pos_tags(data, lib_path):
	tagger = StanfordPOSTagger(lib_path+"models/english-caseless-left3words-distsim.tagger", path_to_jar=lib_path+"stanford-postagger-3.7.0.jar")
	result = np.array([])
	for line in data:
		words = tokenizer(line)
		tags = tagger.tag(words)
		result = np.append(result, tags[-1][1])
	return result

def rename_pos(tag):
	if tag=="NNP" or tag=="NNPS":
		return "PN"
	elif tag=="NN" or tag=="NNS":
		return "CN"
	elif tag=="JJ" or tag=="JJS" or tag=="JJR":
		return "ADJ"
	elif tag=="VB" or tag=="VBD" or tag=="VBG" or tag=="VBN" or tag=="VBP" or tag=="VBZ":
		return "V"
	elif tag=="RB" or tag=="RBS" or tag=="RBR":
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



## MAIN ##

if __name__ == "__main__":

	# Load data

	test_path = "/home/moises/thesis/lambada/lambada-dataset/lambada_test_plain_text.txt"
	with open(test_path, "r", encoding="utf-8") as f:
		test_data = f.readlines()
	test_data = [*map(str.strip, test_data)]

	dev_path = "/home/moises/thesis/lambada/lambada-dataset/lambada_development_plain_text.txt"
	with open(dev_path, "r", encoding="utf-8") as f:
		dev_data = f.readlines()
	dev_data = [*map(str.strip, dev_data)]

	lambada = test_data + dev_data


	# Target word in context or not
	# Yes: 84.67%, No: 15.33% (out of a total of 5153 examples)
	test_context = target_in_context(test_data)
	fd = nltk.FreqDist(test_context)
	fd.tabulate()
	np.save("test_context", test_context)

	# Target word PoS tag (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
	# PN: 44.07%, CN: 40.99%, V: 7.47%, ADJ: 5.12%, ADV: 1.36%, O: 0.9%
	test_pos = pos_tags(test_data, "/home/moises/thesis/stanford-postagger-full-2016-10-31/")
	test_pos = np.array([*map(rename_pos, test_pos)])
	fd = nltk.FreqDist(test_pos)
	fd.tabulate()
	np.save("test_pos", test_pos)


	# Coreference distance (if any)

	# --- WORK IN PROGRESS ---

	# from pycorenlp import StanfordCoreNLP

	# # START THE SERVER: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

	# nlp = StanfordCoreNLP("http://localhost:9000")

	# line="hello my name is moises"

	# output = nlp.annotate(line, properties={
	# 	"annotators": "tokenize,ssplit,pos,lemma,ner,parse,mention,coref",
	# 	"coref.algorithm": "neural",
	# 	"outputFormat": "json"
	# 	})
		
	# # Extract PoS tags (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

	# def get_pos_distribution(file_path):
	# 	pos = []
	# 	with open(file_path, "r", encoding="utf-8") as f:
	# 		for line in f:
	# 			print(nlp.pos_tag(line)[-1])
	# 			pos.append( nlp.pos_tag(line)[-1][1] )
	# 	return Counter(pos)

	# test_pos = get_pos_distribution("/home/moises/thesis/lambada/lambada-dataset/lambada_test_plain_text.txt")

	# dev_pos = get_pos_distribution("/home/moises/thesis/lambada/lambada-dataset/lambada_development_plain_text.txt")