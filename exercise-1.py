from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import operator

#get stop words
stop_words = get_stop_words('en')

#fetch train and test data
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

#we do not want to recalculate idf and we want words and word bi-grams
vectorizer = TfidfVectorizer( use_idf=False, ngram_range=(1, 2), stop_words=stop_words )

#learn idf
trainvec = vectorizer.fit_transform(train.data)

#input document as array
input_document = test.data[:1]

#apply tf
testvec = vectorizer.transform(input_document)

#get words and word bi-grams names
feature_names = vectorizer.get_feature_names()

#tuples with words and word bi-grams names and weight
name_weight = {}

#gather all tuples
#we only need the feature number (only one document)
for i in testvec.nonzero()[1]:
	name_weight[feature_names[i]] = testvec[0, i]

#sort tuples by weight
sorted_name_weight = sorted(name_weight.items(), key=operator.itemgetter(1))

#show top 5 weights (most 'heavy')
for i in range(0,5):
	print sorted_name_weight[i][0] + ' : ' + str(sorted_name_weight[i][1])

