#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/17/2018
## DUE: 1/24/2018
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import matplotlib.pyplot as plt
import numpy as np
import syllables
from nltk.corpus import wordnet as wn
from sklearn.ensemble import RandomForestClassifier

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
	## YOUR CODE HERE...
	tp=0
	fp=0
	for i in range(len(y_pred)):
		if y_pred[i]==1 and y_true[i]==1:
			tp+=1
		if y_pred[i]==1 and y_true[i]==0:
			fp+=1
	if tp==0 and fp==0:
		return 1
	precision=1.0*tp/(1.0*(tp+fp))
	return precision
	
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
	## YOUR CODE HERE...
	tp=0
	fn=0
	for i in range(len(y_pred)):
		if y_pred[i]==1 and y_true[i]==1:
			tp+=1
		if y_pred[i]==0 and y_true[i]==1:
			fn+=1
	if tp==0 and fn==0:
		return 1
	recall=1.0*tp/(1.0*(tp+fn))
	return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
	## YOUR CODE HERE...
	precision=get_precision(y_pred,y_true)
	recall=get_recall(y_pred,y_true)
	if precision==0 and recall==0:
		return 0
	fscore=2.0*precision*recall/(precision+recall)
	return fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
	words = []
	labels = []   
	with open(data_file, 'rt', encoding="utf8") as f:
		i = 0
		for line in f:
			if i > 0:
				line_split = line[:-1].split("\t")
				words.append(line_split[0].lower())
				labels.append(int(line_split[1]))
			i += 1
	return words, labels

def load_file1(data_file):
	words = []
	labels = []   
	with open(data_file, 'rt', encoding="utf8") as f:
		i = 0
		for line in f:
			if i > 0:
				line_split = line[:-1].split("\t")
				words.append(line_split[0].lower())
			i += 1
	return words, labels


def load_file2(data_file):
	words = []
	labels = []   
	with open(data_file, 'rt', encoding="utf8") as f:
		i = 0
		for line in f:
			if i > 0:
				line_split = line.split("\t")
				words.append(line_split[1].lower())
				labels.append(int(line_split[3]))
			i += 1
	return words, labels


### 2.1: A very simple baseline

## Labels every word complex
def all_complex(data_file):
	## YOUR CODE HERE...
	words, y_true=load_file(data_file)
	y_pred=[1.0]*len(y_true)
	precision=get_precision(y_pred,y_true)
	recall=get_recall(y_pred,y_true)
	fscore=get_fscore(y_pred,y_true)
	performance = [precision, recall, fscore]
	return performance


### 2.2: Word length thresholding

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
	## YOUR CODE HERE
	words, y_true=load_file(training_file)
	maxn=0
	precisions=[]
	recalls=[]
	for i in range(len(words)):
		if maxn<len(words[i]):
			maxn=len(words[i])
	index=-1
	max_fscore=-1
	for i in range(maxn):
		y_pred=[]
		for j in range(len(words)):
			if len(words[j])<i:
				y_pred.append(0)
			else:
				y_pred.append(1)
		precision=get_precision(y_pred,y_true)
		recall=get_recall(y_pred,y_true)
		fscore=get_fscore(y_pred,y_true)
		if max_fscore<fscore:
			max_fscore=fscore
			index=i
		precisions.append(precision)
		recalls.append(recall)

	#print (precisions)
	print (index)
	plt.step(recalls,precisions,color='blue')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	#plt.ylim([0.00,1.05])
	#plt.xlim([0.00,1.05])
	plt.title('2 class precision-recall curve')

	y_pred=[]
	for i in range(len(words)):
		if len(words[i])<index:
			y_pred.append(0)
		else:
			y_pred.append(1)

	tprecision=get_precision(y_pred,y_true)
	trecall=get_recall(y_pred,y_true)
	tfscore=get_fscore(y_pred,y_true)

	y_pred=[]
	words,y_true=load_file(development_file)
	for i in range(len(words)):
		if len(words[i])<index:
			y_pred.append(0)
		else:
			y_pred.append(1)
	dprecision=get_precision(y_pred,y_true)
	drecall=get_recall(y_pred,y_true)
	dfscore=get_fscore(y_pred,y_true)

	training_performance = [tprecision, trecall, tfscore]
	development_performance = [dprecision, drecall, dfscore]
	return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt',encoding='utf-8') as f: 
	   for line in f:
		   token, count = line.strip().split('\t') 
		   if token[0].islower(): 
			   counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts):
#	words, y_true=load_file(training_file)
#	maxn=0
#	precisions=[]
#	recalls=[]
#	for i in range(len(words)):
#		if maxn<len(words[i]):
#			maxn=len(words[i])
#	index=-1
#	max_fscore=-1
#	for i in range(maxn):
#		y_pred=[]
#		for j in range(len(words)):
#			if len(words[j])<i:
#				y_pred.append(0)
#			else:
#				y_pred.append(1)
#		precision=get_precision(y_pred,y_true)
#		recall=get_recall(y_pred,y_true)
#		fscore=get_fscore(y_pred,y_true)
#		if max_fscore<fscore:
#			max_fscore=fscore
#			index=i
#		precisions.append(precision)
#		recalls.append(recall)
#
#	#print (precisions)
#	#print (recalls)
#	plt.plot(recalls,precisions,color='blue')

	## YOUR CODE HERE
	words,y_true=load_file(training_file)
	maxn=0
	precisions=[]
	recalls=[]
	freq=[]

	for i in range(len(words)):
		freq.append(counts[words[i]])

	threshold=-1
	max_fscore=-1

	for i in range(0,int(1e+10),int(1e+6)):
		#print ("h")
		print (i)
		y_pred=[]
		
		for j in range(len(words)):
			if counts[words[j]]<i:
				y_pred.append(1)
			else:
				y_pred.append(0)

		precision=get_precision(y_pred,y_true)
		recall=get_recall(y_pred,y_true)
		fscore=get_fscore(y_pred,y_true)

		if max_fscore<fscore:
			max_fscore=fscore
			threshold=i

		precisions.append(precision)
		recalls.append(recall)

	print (threshold)
	plt.plot(recalls,precisions,color='red')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	#plt.ylim([0.00,1.05])
	#plt.xlim([0.00,1.05])
	plt.title('2 class precision-recall curve')

	y_pred=[]
	for i in range(len(words)):
		if counts[words[i]]<threshold:
			y_pred.append(1)
		else:
			y_pred.append(0)

	tprecision=get_precision(y_pred,y_true)
	trecall=get_recall(y_pred,y_true)
	tfscore=get_fscore(y_pred,y_true)

	y_pred=[]
	words,y_true=load_file(development_file)
	for i in range(len(words)):
		if counts[words[i]]<threshold:
			y_pred.append(1)
		else:
			y_pred.append(0)
	dprecision=get_precision(y_pred,y_true)
	drecall=get_recall(y_pred,y_true)
	dfscore=get_fscore(y_pred,y_true)

	training_performance = [tprecision, trecall, tfscore]
	development_performance = [dprecision, drecall, dfscore]
	return training_performance, development_performance

### 2.4: Naive Bayes
		
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
	## YOUR CODE HERE

	words,y_true=load_file(training_file)
	feat1=[]
	feat2=[]
	for i in range(len(words)):
		feat1.append(len(words[i]))
		feat2.append(counts[words[i]])
	mean1=np.mean(feat1)
	mean2=np.mean(feat2)
	std1=np.std(feat1)
	std2=np.std(feat2)
	Xtrain=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2))
	from sklearn.naive_bayes import GaussianNB
	clf=GaussianNB()
	clf.fit(Xtrain,y_true)
	y_pred=clf.predict(Xtrain)

	tprecision=get_precision(y_pred,y_true)
	trecall=get_recall(y_pred,y_true)
	tfscore=get_fscore(y_pred,y_true)

	words,y_true=load_file(development_file)
	feat1=[]
	feat2=[]
	for i in range(len(words)):
		feat1.append(len(words[i]))
		feat2.append(counts[words[i]])

	Xtest=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2))
	y_pred=clf.predict(Xtest)

	dprecision=get_precision(y_pred,y_true)
	drecall=get_recall(y_pred,y_true)
	dfscore=get_fscore(y_pred,y_true)
	
	training_performance = [tprecision, trecall, tfscore]
	development_performance = [dprecision, drecall, dfscore]
	return training_performance, development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
	## YOUR CODE HERE    
	words,y_true=load_file(training_file)
	feat1=[]
	feat2=[]
	for i in range(len(words)):
		feat1.append(len(words[i]))
		feat2.append(counts[words[i]])
	mean1=np.mean(feat1)
	mean2=np.mean(feat2)
	std1=np.std(feat1)
	std2=np.std(feat2)
	Xtrain=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2))
	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression()
	clf.fit(Xtrain,y_true)
	y_pred=clf.predict(Xtrain)

	tprecision=get_precision(y_pred,y_true)
	trecall=get_recall(y_pred,y_true)
	tfscore=get_fscore(y_pred,y_true)

	words,y_true=load_file(development_file)
	feat1=[]
	feat2=[]
	for i in range(len(words)):
		feat1.append(len(words[i]))
		feat2.append(counts[words[i]])

	Xtest=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2))
	y_pred=clf.predict(Xtest)

	dprecision=get_precision(y_pred,y_true)
	drecall=get_recall(y_pred,y_true)
	dfscore=get_fscore(y_pred,y_true)
	
	training_performance = [tprecision, trecall, tfscore]
	development_performance = [dprecision, drecall, dfscore]
	return training_performance, development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE
#def own_classifier(training_file,development_file,test_file1,counts):
def own_classifier(training_file,development_file,test_file1,extra_file,counts):
	words,y_true1=load_file(training_file)
	words1,y_true2=load_file2(extra_file)
	words.extend(words1)
	y_true1.extend(y_true2)
	feat1=[]
	feat2=[]
	feat3=[]
	feat4=[]

	for i in range(len(words)):
		#print (i)
		#print (words)
		feat1.append(len(words[i]))
		if words[i] not in counts:
			counts[words[i]]=1
		feat2.append(counts[words[i]])
		feat3.append(syllables.count_syllables(words[i]))
		feat4.append(len(wn.synsets(words[i])))

	mean1=np.mean(feat1)
	mean2=np.mean(feat2)
	mean3=np.mean(feat3)
	mean4=np.mean(feat4)

	std1=np.std(feat1)
	std2=np.std(feat2)
	std3=np.std(feat3)
	std4=np.std(feat4)

	Xtrain=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2,(feat3-mean3)/std3, (feat4-mean4)/std4))
	
	best_fscore=-1
	dep=-1
	est=-1
	
	for a in range(1,12):
		for b in range(500,501,1):
			clf = RandomForestClassifier(max_depth=a,n_estimators=b,criterion='entropy',bootstrap=False)
#			from sklearn.neural_network import MLPClassifier
#			clf = MLPClassifier(alpha=1e-2, hidden_layer_sizes=(5, 2), random_state=1)

	# from sklearn.svm import SVC
	# clf = SVC(C=1,tol=1e-9,gamma=0.10)

	# from sklearn import tree
	# clf = tree.DecisionTreeClassifier(max_depth=4)
	
			clf.fit(Xtrain,y_true1)
			y_pred=clf.predict(Xtrain)

			tprecision=get_precision(y_pred,y_true1)
			trecall=get_recall(y_pred,y_true1)
			tfscore=get_fscore(y_pred,y_true1)

			words,y_true=load_file(development_file)
			feat1=[]
			feat2=[]
			feat3=[]
			feat4=[]
			for i in range(len(words)):
				feat1.append(len(words[i]))
				feat2.append(counts[words[i]])
				feat3.append(syllables.count_syllables(words[i]))
				feat4.append(len(wn.synsets(words[i])))

			#print (len(feat1))
			#print (len(feat2))
			#print (len(feat3))
			#print (len(feat4))

			Xtest=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2,(feat3-mean3)/std3, (feat4-mean4)/std4))
			y_pred=clf.predict(Xtest)

			dprecision=get_precision(y_pred,y_true)
			drecall=get_recall(y_pred,y_true)
			dfscore=get_fscore(y_pred,y_true)
		
			training_performance = [tprecision, trecall, tfscore]
			development_performance = [dprecision, drecall, dfscore]
			if best_fscore<dfscore:
				best_fscore=dfscore
				dep=a
				est=b

	print (best_fscore)
	print (dep)
	print (est)

	clf = RandomForestClassifier(max_depth=dep,n_estimators=est,criterion='entropy',bootstrap=False)
#	clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#	Xtrain=np.vstack((Xtrain,Xtest))
#	y_true1.append(y_true)
	clf.fit(Xtrain,y_true1)

	words,y_true=load_file1(test_file1)
	feat1=[]
	feat2=[]
	feat3=[]
	feat4=[]
	for i in range(len(words)):
	    feat1.append(len(words[i]))
	    feat2.append(counts[words[i]])
	    feat3.append(syllables.count_syllables(words[i]))
	    feat4.append(len(wn.synsets(words[i])))

	#print (len(feat1))
	#print (len(feat2))
	#print (len(feat3))
	#print (len(feat4))

	Xtest=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2,(feat3-mean3)/std3, (feat4-mean4)/std4))
	y_pred=clf.predict(Xtest)
	#y_pred=[int(x) for x in y_pred]
	with open('test_labels.txt','w') as f:
	    y_pred=list(map(lambda a: str(a)+'\n', y_pred))
	    f.writelines(y_pred)
        
	return training_performance, development_performance

def test_predictions(training_file,test_file1,counts):
    words,y_true=load_file(training_file)
    feat1=[]
    feat2=[]
    feat3=[]
    feat4=[]

    for i in range(len(words)):
        feat1.append(len(words[i]))
        feat2.append(counts[words[i]])
        feat3.append(syllables.count_syllables(words[i]))
        feat4.append(len(wn.synsets(words[i])))

    mean1=np.mean(feat1)
    mean2=np.mean(feat2)
    mean3=np.mean(feat3)
    mean4=np.mean(feat4)

    std1=np.std(feat1)
    std2=np.std(feat2)
    std3=np.std(feat3)
    std4=np.std(feat4)

    Xtrain=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2,(feat3-mean3)/std3, (feat4-mean4)/std4))
    
    clf = RandomForestClassifier(max_depth=7,n_estimators=1000, criterion='entropy')
    clf.fit(Xtrain,y_true)
    
    words,y_true=load_file(test_file1)
    feat1=[]
    feat2=[]
    feat3=[]
    feat4=[]
    for i in range(len(words)):
        feat1.append(len(words[i]))
        feat2.append(counts[words[i]])
        feat3.append(syllables.count_syllables(words[i]))
        feat4.append(len(wn.synsets(words[i])))

    print (len(feat1))
    print (len(feat2))
    print (len(feat3))
    print (len(feat4))

    Xtest=np.column_stack(((feat1-mean1)/std1,(feat2-mean2)/std2,(feat3-mean3)/std3, (feat4-mean4)/std4))
    y_pred=clf.predict(Xtest)
    #y_pred=[int(x) for x in y_pred]
    s=np.column_stack((words,y_true,y_pred))
    import pandas as pd
    df=pd.DataFrame(s)
    df.to_csv('f.csv')
    #np.savetxt("FILENAME.csv", s, delimiter=",")
#    with open('test_labels.txt','w') as f:
#        y_pred=list(map(lambda a: str(a)+'\n', y_pred))
#        f.writelines(y_pred)
    #np.savetxt('test_labels.txt',y_pred)
    
if __name__ == "__main__":
	training_file = "data/complex_words_training.txt"
	development_file = "data/complex_words_development.txt"
	test_file = "data/complex_words_test_unlabeled.txt"

	train_data = load_file(training_file)
	
	ngram_counts_file = "ngram_counts.txt.gz"
	counts = load_ngram_counts(ngram_counts_file)
