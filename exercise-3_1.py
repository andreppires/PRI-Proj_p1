from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import operator
import nltk

# get stop words
stop_words = get_stop_words('en')

# user must select document to test min=1!
docTest = 1

#fetch train and test data
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

# In this case we only wanna some patterns of tokens
#   JJ-NN-IN
#   JJ-NN
#   NN
#   JJ-adjective    IN-preposition  NN-noun
newtrain=[]
flag1=False
flag2=False
for doc in train.data[:100]:
    newwords = nltk.word_tokenize(doc)
    token = nltk.pos_tag(newwords)
    newdoc=''
    for i in range(0, len(token)):
        if flag1:
            continue
        if flag2:
            continue
        candidate = ''
        if i < len(token) - 2 and token[i][1] == 'JJ' and token[i + 1][1] == 'NN' and token[i + 2][1] == 'IN':
            candidate = (token[i][0] + ' ' + token[i + 1][0] + ' ' + token[i + 2][0])
            flag1 = True
            flag2=True
        elif i < len(token) - 1 and token[i][1] == 'JJ' and token[i + 1][1] == 'NN':
            candidate = (token[i][0] + ' ' + token[i + 1][0])
            flag1 = True
        elif token[i][1] == 'NN':
            candidate = token[i][0]
        if candidate !='':
            newdoc=newdoc+' '+candidate
    newtrain.append(newdoc)
    flag1=False
    flag2=False

# we do not want to recalculate idf and we want words and word bi-grams or tri-gramas
vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,3), stop_words=stop_words)

# learn idf
trainvec = vectorizer.fit_transform(newtrain)

# input document as array
input_document = test.data[:1]

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