# from stanfordcorenlp import StanfordCoreNLP
# from collections import Counter

# nlp = StanfordCoreNLP("/home/moises/thesis/stanford-corenlp-full-2016-10-31")
	
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

# NOTE: Everything is lowercased so I won't get Proper Nouns :( 


import nltk
import numpy as np


## FUNC'S DEFINITION ##

def target_in_context(data):
	lemma = nltk.wordnet.WordNetLemmatizer()
	result = np.array([])
	for line in data:
		words = line.split()
		words = [ lemma.lemmatize(word) for word in words ]
		context = words[:-1]
		target = words[-1]
		result = np.append(result, target in context)
	return result

## MAIN ##

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
test_context = target_in_context(test_data)
np.save("test_context", test_context)


# Target word PoS tag


# Coreference distance (if any)

