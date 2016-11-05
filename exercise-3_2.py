from sklearn.datasets import fetch_20newsgroups
from stop_words import get_stop_words
import operator
import nltk
import math

# calculate idf(t)
def calculate_idf(doc_candidates):
    idf = {}  # idf(t)
    N = len(doc_candidates.keys()) # number of docs
    nt = {} # n(t)
    
    # calculate n(t)
    # for each doc
    for doc in doc_candidates.keys():
        tokens = nltk.word_tokenize(doc)
        # for each candidate from doc
        for candidate in all_candidates[doc]:
            if candidate in tokens:
                if candidate in nt.keys():
                    nt[candidate]+=1
                else:
                    nt[candidate]=1
    # calculate idf
    for candidate in nt.keys():
        idf[candidate]=math.log((N-nt[candidate]+0.5)/(nt[candidate]+0.5))
    
    return idf

# calculate f(t, D)
def calculate_tf(term, doc):
    nt = 0  # number of occurences of term in doc
    
    tokens = nltk.word_tokenize(doc)
    N = len(tokens) # lenght of doc
    
    if term in tokens:
        nt += 1
    
    return nt / N # term frequency in doc

# calculate doc lenght
def calculate_len(doc):
    return len(nltk.word_tokenize(doc))

# calculate average document size
def calculate_avg(docs):
    num_docs = len(docs)
    lenghts = 0
    
    for doc in docs:
        lenghts += calculate_len(doc)

    return lenghts / num_docs # average doc lenght

# calculate bm25 scores
# accepts a dictionary with ONE doc, multiple candidates and average doc size
def calculate_bm25(doc_candidates, idf, avg_doc_size):
    score = {}
    k1 = 1.2
    b = 0.75
    
    # for each doc
    for doc in doc_candidates.keys():
        # for each candidate from doc
        for candidate in all_candidates[doc]:
            tf = calculate_tf(candidate, doc)
            if candidate in idf.keys():
                score[str(candidate)] = idf[candidate] * ((tf * (k1 + 1)) / (tf + (k1 * (1 - b + (b * (calculate_len(doc) / avg_doc_size))))))
            else:
                score[str(candidate)] = 0
    
    return score

# get stop words
stop_words = get_stop_words('en')

# user must select document to test min=1!
docTest = 1

#fetch train and test data
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

data = train.data[:100]

# In this case we only wanna some patterns of tokens
#   JJ-NN-IN
#   JJ-NN
#   NN
#   JJ-adjective    IN-preposition  NN-noun
newtrain=[]
flag1=False
flag2=False
all_candidates={}

for doc in data:
    newwords = nltk.word_tokenize(doc)
    token = nltk.pos_tag(newwords)
    newdoc=''
    doc_candidates=[]
    for i in range(0, len(token)):
        if flag1:
            continue
        if flag2:
            continue
        candidate = ''
        if i < len(token) - 2 and token[i][1] == 'JJ' and token[i + 1][1] == 'NN' and token[i + 2][1] == 'IN':
            candidate = (token[i][0] + ' ' + token[i + 1][0] + ' ' + token[i + 2][0])
            flag1 = True
            flag2 = True
        elif i < len(token) - 1 and token[i][1] == 'JJ' and token[i + 1][1] == 'NN':
            candidate = (token[i][0] + ' ' + token[i + 1][0])
            flag1 = True
        elif token[i][1] == 'NN':
            candidate = token[i][0]
        if candidate != '':
            newdoc=newdoc+' '+candidate
            doc_candidates.append(candidate)
    newtrain.append(newdoc)
    all_candidates[doc]=doc_candidates
    flag1=False
    flag2=False

idf = calculate_idf(all_candidates)

avg_doc_size = calculate_avg(all_candidates.keys())

# create doc test
doc_test = {}
for doc in all_candidates.keys():
    doc_test[doc] = all_candidates[doc]
    break;

#print test
score = calculate_bm25(doc_test, idf, avg_doc_size)

# sort tuples by weight
sorted_name_weight = sorted(score.items(), key=operator.itemgetter(1))

# show top 5 weights (most 'heavy')
for i in range(0, 5):
    print sorted_name_weight[i][0] + ' : ' + str(sorted_name_weight[i][1])