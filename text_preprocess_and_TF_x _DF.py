from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import random



data = []
letter = ["A","B","C","D","E","F","G","H","I","J"]
min_df = 2 # if a word emerges in less than 2 doc, than, not be considered
max_df = 0.7 #float,percentage
lowercase = True # could change

for i in range(1,101):
	padnumber = '{:04d}'.format(i) #000x
	for fileletter in letter:
		file = open("/Users/chaowang/Desktop/Descubrimiento_De_Información_En_Textos/7/BankSearch-reduced/{}{}.txt".format(fileletter,padnumber), encoding='cp1252')
		mystr = file.read()
		# index = mystr.find("<html>") #extract text from HTML
		# mystr = mystr[index:]
		# soup = BeautifulSoup(mystr, 'html.parser')#parse a html file, analisis as html file
		t = mystr # soup.get_text()
		data.append((t, fileletter))#reprepare for shuffle	# every element is a tuple

random.shuffle(data)
print('Num of doc: ',len(data)) # for verification
train_data = ([], [])#(texts, classes)	#[0],[1]	#tuple	# every element is a list
test_data = ([], [])
for i in range(1000):
	if i < 700:
		train_data[0].append(data[i][0])
		train_data[1].append(data[i][1])
	else:
		test_data[0].append(data[i][0])
		test_data[1].append(data[i][1])

print('Num of doc: ',len(train_data[0])) # for verification
print('Num of doc: ',len(test_data[0]))


def Stemmer(Data):
    for i in range(len(Data)):
        Data[i] = ' '.join(map(SnowballStemmer('english').stem, CountVectorizer().build_analyzer()(Data[i]))) # to stem every word

def Counter(train, test):
    Stemmer(train)
    Stemmer(test)
    #Convert a collection of text documents to a matrix of token counts:
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, lowercase=lowercase, stop_words=text.ENGLISH_STOP_WORDS) ## sklearn stopwords
    X_train_counts = count_vect.fit_transform(train)
    print('Shape of train =',X_train_counts.shape)
    X_test_counts = count_vect.transform(test)
    print('Shape of test =',X_test_counts.shape) # builds vectors
    # the same vocabulary
    print(count_vect.get_feature_names()[1947]) # 1947 for verfication: what is the 1974 attribute word in the metrix
    return X_train_counts, X_test_counts, count_vect.get_feature_names()

train_counts, test_counts, vocabulary = Counter(train_data[0], test_data[0]) # text converts to metrix

from sklearn.feature_extraction.text import TfidfTransformer # convert a count metrix to tfidf metrix
def tfidf(X_train_counts, X_test_counts):
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) #fit to data, then transform it. (find a funtion that suits those datas)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print(X_train_tfidf.shape)
    print(X_test_tfidf.shape)
    return X_train_tfidf, X_test_tfidf

train_tfidf, test_tfidf = tfidf(train_counts, test_counts)

train_arff = open("/Users/chaowang/Desktop/Descubrimiento_De_Información_En_Textos/7/BankSearch-reduced/train.arff", "w")
test_arff = open("/Users/chaowang/Desktop/Descubrimiento_De_Información_En_Textos/7/BankSearch-reduced/test.arff", "w")  # Open it by writing

train_arff.write("@RELATION train\n\n")
test_arff.write("@RELATION train\n\n")

train_arff.write("@ATTRIBUTE Label {A,B,C,D,E,F,G,H,I,J}\n")
test_arff.write("@ATTRIBUTE Label {A,B,C,D,E,F,G,H,I,J}\n")

for i in range(train_tfidf.shape[1]):
	train_arff.write("@ATTRIBUTE {} numeric\n".format(vocabulary[i]))
	test_arff.write("@ATTRIBUTE {} numeric\n".format(vocabulary[i]))

train_arff.write("\n@DATA\n")
test_arff.write("\n@DATA\n")

for i in range(700):
	train_arff.write("{},{}\n".format(train_data[1][i], ','.join(str(v) for v in train_tfidf.toarray()[i])))
for i in range(300):
	test_arff.write("{},{}\n".format(test_data[1][i], ','.join(str(v) for v in test_tfidf.toarray()[i])))


	


