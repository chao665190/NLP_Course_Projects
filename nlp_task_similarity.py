from gensim.models import KeyedVectors
import nltk
from nltk import word_tokenize
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

brown_ic = wordnet_ic.ic('ic-brown.dat')
genesis_ic = wn.ic(genesis, False, 0.0)
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
ic = semcor_ic

cache_map = {}
model = None


def EliminateToken(tokens):
	tuples = nltk.pos_tag(tokens)
	tokens = []
	for token, pos in tuples:
		if pos != "CC" and pos != "TO" and pos != "DT":
			tokens.append(token)
	return tokens

def Stemmer(tokens):
	new_tokens = []
	porter = nltk.PorterStemmer()
	for t in tokens:
		t = porter.stem(t)
		if len(t) != 1 or t.isalnum():
			new_tokens.append(t)
	return new_tokens


def FindMaxSimForPos(w, wi, pos):
	w_synsets = wn.synsets(w, pos)
	wi_synsets = wn.synsets(wi, pos)
	max_sim = 0
	for w_synset in w_synsets:
		for wi_synset in wi_synsets:
			#if w_synset.pos() not in ic or wi_synset.pos() not in ic:
				#continue
			sim = wn.path_similarity(w_synset, wi_synset)
			if sim == 1:
				return 1
			if sim != None:
				max_sim = max(max_sim, sim)
	return max_sim			

def FindMaxSim(w, wi):
	if wi == w:
		return 1
	if (w, wi) in cache_map:#共8000多个句子，之前找过的单词对存入cache，节省计算量
		return cache_map[(w, wi)]
	max_sim = 0
	max_sim = max(max_sim, FindMaxSimForPos(w, wi, wn.NOUN))
	if max_sim == 1:
		cache_map[(w, wi)] = 1
		return 1
	max_sim = max(max_sim, FindMaxSimForPos(w, wi, wn.ADJ))
	if max_sim == 1:
		cache_map[(w, wi)] = 1
		return 1
	max_sim = max(max_sim, FindMaxSimForPos(w, wi, wn.ADV))
	if max_sim == 1:
		cache_map[(w, wi)] = 1
		return 1
	max_sim = max(max_sim, FindMaxSimForPos(w, wi, wn.VERB))
	cache_map[(w, wi)] = max_sim
	return max_sim


def FindMaxSimInSentence(w, t):
	max_sim = 0
	for wi in t:
		max_sim = max(max_sim, FindMaxSim(w, wi))
		if max_sim == 1:
			break
	
	return max_sim

def BuildWordNetBasedVectors(s1, s2):
	t1 = set(s1)
	t2 = set(s2)

	t = sorted(t1.union(t2))

	v1 = []
	v2 = []
	for w in t:
		v1.append(FindMaxSimInSentence(w, t1))
		v2.append(FindMaxSimInSentence(w, t2))	
	return np.array(v1), np.array(v2)


def BuildWord2VecBasedVectors(s1, s2):
	v1 = np.zeros(model.vector_size)
	v2 = np.zeros(model.vector_size)
	for word in s1:
		if word in model.vocab:
			v1 += model.word_vec(word)

	for word in s2:
		if word in model.vocab:
			v2 += model.word_vec(word)
	return v1, v2


f = open('/Users/chaowang/Desktop/nlp/final_task/stsbenchmark/sts-train.csv')

scorelist = []
sentencetokens = []

pattern = r'''(?x)     # set flag to allow verbose regexps
		\w+       # words with optional internal hyphens
	 '''

wnl = nltk.WordNetLemmatizer()

use_word2vec = False
eliminate_token = True
lower_case = True
lemmatize_token = True

if use_word2vec:
	model = KeyedVectors.load_word2vec_format('/Users/chaowang/Desktop/nlp/final_task/GoogleNews-vectors-negative300.bin',binary=True)

for line in f:
	line = line.strip()
	elements = line.split("\t")
	
	scorelist.append(float(elements[4]))

	sentence1token = nltk.regexp_tokenize(elements[5], pattern)
	sentence2token = nltk.regexp_tokenize(elements[6], pattern)
	if eliminate_token:
		sentence1token = EliminateToken(sentence1token)
		sentence2token = EliminateToken(sentence2token)
	if lower_case:
		sentence1token = [letters.lower() for letters in sentence1token]
		sentence2token = [letters.lower() for letters in sentence2token]
	if lemmatize_token:
		sentence1token = [wnl.lemmatize(t) for t in sentence1token]
		sentence2token = [wnl.lemmatize(t) for t in sentence2token]

	sentencetokens.append([sentence1token, sentence2token])

simlist = []
for elem in sentencetokens:
	if use_word2vec:
		v1, v2 = BuildWord2VecBasedVectors(elem[0], elem[1])
	else:
		v1, v2 = BuildWordNetBasedVectors(elem[0], elem[1])
	
	cos_sim = cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))[0][0]
	simlist.append(cos_sim)

print(pearsonr(simlist, scorelist))

