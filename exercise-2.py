from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from stop_words import get_stop_words
import operator
import os
import time

# get stop words
stop_words = get_stop_words('en')

# user must select document to test min=1!
docTest = 1

# fetch train and test data
test = fetch_20newsgroups(subset='test')

# fetching train data/collenction from disk
# using wiki20 collection from https://github.com/zelandiya/keyword-extraction-datasets
train = []
for file in os.listdir("wiki20/documents"):
    if file.endswith(".txt"):
        with open("wiki20/documents/" + file, 'r') as myfile:
            train.append(myfile.read().replace('\n', ''))

# we do not want to recalculate idf and we want words and word bi-grams
vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1, 2), stop_words=stop_words)

# learn idf
trainvec = vectorizer.fit_transform(train)

# input document as array
input_document = train[docTest - 1:docTest]

# apply tf
testvec = vectorizer.transform(input_document)

# get words and word bi-grams names
feature_names = vectorizer.get_feature_names()

# tuples with words and word bi-grams names and weight
name_weight = {}

# gather all tuples
# we only need the feature number (only one document)
for i in testvec.nonzero()[1]:
    name_weight[feature_names[i]] = testvec[0, i]

# sort tuples by weight
sorted_name_weight = sorted(name_weight.items(), key=operator.itemgetter(1))

# show top 5 weights (most 'heavy')
for i in range(0, 5):
    print sorted_name_weight[i][0] + ' : ' + str(sorted_name_weight[i][1])

# fetching result data/collection from disk
# using wiki20 collection from https://github.com/zelandiya/keyword-extraction-datasets
humanResult = []
file = os.listdir("wiki20/teams/team1")
with open("wiki20/teams/team1/" + file[docTest - 1], 'r') as myfile:
    for line in myfile:
        aux = line.split(":")
        aux = aux[1].split("(")
        aux[0] = aux[0][1:]
        if ((aux[0].endswith(" ")) or (aux[0].endswith("\n"))):
            aux[0] = aux[0][:-1]
        humanResult.append(aux[0])

tdidfresult = []
for i in sorted_name_weight[: len(humanResult)]:
    tdidfresult.append(str(i[0]))
print "PREDICT"
print tdidfresult
print "REAL"
print humanResult

# print "SCORE"
# time.sleep(1)
# recall
# r=recall_score(humanResult, tdidfresult, average=None)
# print r
# fi_score
# f=f1_score(humanResult, tdidfresult, average=None)
# print f
# precision
# p=precision_score(humanResult, tdidfresult, average=None)
# print p
# APS
# a=average_precision_score(humanResult, tdidfresult, average=None)
# print a

# precision = (true intersect pred)/(#true)

# recall = (true intersect pred)/(#pred)

# f1Score = (2*precision*recall)/(precision+recall)

intersection= set(tdidfresult).intersection(humanResult)

prediction = len(intersection)/len(humanResult)
print "prediction= "+ str(prediction)

recall = len(intersection)/len(tdidfresult)
print "recall= "+str(recall)

if (prediction+recall ==0):
    f1Score=0
else:
    f1Score= (2*prediction*recall)/(prediction+recall)
print "F1 Score= "+ str(f1Score)