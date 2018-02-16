# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:15:15 2018

@author: sharvin
"""

from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans
from nltk.stem.porter import *
from nltk.stem.snowball import *

# This maps from word  -> list of candidates
word2cands = {}

# This maps from word  -> number of clusters
word2num = {}

# Read the words file.
with open(r"./data/dev_input.txt") as f:
    for line in f:
        word, numclus, cands = line.split(" :: ")
        cands = cands.split()
        word2num[word] = int(numclus)
        word2cands[word] = cands

# Load cooccurrence vectors (question 2)
#vec = KeyedVectors.load_word2vec_format(r"C:\Users\sharvin\Downloads\CIS 530 HW4\coocvec-250mostfreq-window-1.vec")
# Load dense vectors (uncomment for question 3)
vec = KeyedVectors.load_word2vec_format(r"./data/GoogleNews-vectors-negative300.filter")
#vec = KeyedVectors.load_word2vec_format(r"paragram_vectors.txt")
f = open(r'test_output_leaderboard.txt', 'w')        
for word in word2cands:
    cands = word2cands[word]
    numclusters = word2num[word]

    # TODO: get word vectors from vec
    # Cluster them with k-means
    # Write the clusters to file.
    p=[.99]+list(.01/249*np.ones((249,)))
    vecs=[]
    for cand in cands:
        if cand in vec:
            vecs.append(vec[cand])
        else:
            stemmer = PorterStemmer()
            cand1=stemmer.stem(cand)
            if cand1 in vec:
                vecs.append(vec[cand1])
            else:
                stemmer = SnowballStemmer("english")
                cand=SnowballStemmer("english").stem(cand)
                if cand in vec:
                    vecs.append(vec[cand])
                else:
                    cand=SnowballStemmer("porter").stem(cand)
                    if cand in vec:
                        vecs.append(vec[cand])
                    else:
                        #a=np.random.choice(np.arange(250), size=(300,),p=p)
                        a=np.random.choice(np.arange(50), size=(300,))
                        #print (type(a))
                        a=a/(np.sum(a)+1e-6)
                        vecs.append(a)
    kmeans = KMeans(n_clusters=numclusters).fit(vecs)
    cluster = [[] for x in range(numclusters)]
#    print(cluster)
#    print(numclusters)
#    print(kmeans.labels_)

    for i in range((len(kmeans.labels_))):
        cluster_no = kmeans.labels_[i]
        cand = cands[i]
        print(cluster_no, cand)
        cluster[cluster_no].append(cand)

    for i in range((numclusters)):
        print(word + " :: " + str(i + 1) + " :: " + " ".join(x for x in cluster[i])) 
        f.write(word + " :: " + str(i + 1) + " :: " + " ".join(x for x in cluster[i]) + '\n')

f.close()


    # TODO: get word vectors from vec
    # Cluster them with k-means
    # Write the clusters to file.

