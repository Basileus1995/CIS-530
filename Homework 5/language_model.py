from collections import *
from random import random
import numpy as np
import codecs

def train_char_lm(fname, order=4, add_k=1):
  ''' Trains a language model.

  This code was borrowed from 
  http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

  Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED

  Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
  '''

  # TODO: Add your implementation of add-k smoothing.

  data = codecs.open(fname, 'r', encoding='utf8', errors='replace').read()
  lm = defaultdict(Counter)
  pad = "~" * order
  data = pad + data
  vocab=list(set(data))
  length=float(len(vocab))
  
  
  add_k_counts=defaultdict(Counter)
  
  for i in range(len(data)-order):
    history, char = data[i:i+order], data[i+order]
    if history in add_k_counts:
        add_k_counts[history][char]+=1
    else:
        for z in range(len(vocab)):
            add_k_counts[history][vocab[z]]=0
        add_k_counts[history][char]=1
    lm[history][char]+=1
  def normalize(counter):
    s = float(sum(counter.values()))
    return [(c,(cnt+add_k)/(s+add_k*length)) for c,cnt in counter.items()]
    return [(c,cnt/s) for c,cnt in counter.items()]
  outlm = {hist:normalize(chars) for hist, chars in add_k_counts.items()}
  return outlm


def generate_letter(lm, history, order):
  ''' Randomly chooses the next letter using the language model.
  
  Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of text at least 'order' long.
    order: The length of the n-grams in the language model.
    
  Returns: 
    A letter
  '''
  
  history = history[-order:]
  dist = lm[history]
  x = random()
  for c,v in dist:
    x = x - v
    if x <= 0: return c
    
    
def generate_text(lm, order, nletters=500):
  '''Generates a bunch of random text based on the language model.
  
  Inputs:
  lm: The output from calling train_char_lm.
  history: A sequence of previous text.
  order: The length of the n-grams in the language model.
  
  Returns: 
    A letter  
  '''
  history = "~" * order
  out = []
  for i in range(nletters):
    c = generate_letter(lm, history, order)
    print (c)
    history = history[-order:] + c
    out.append(c)
  return "".join(out)

def perplexity(test_filename, lm, order=4):
	'''Computes the perplexity of a text file given the language model.
  
	Inputs:
		test_filename: path to text file
		lm: The output from calling train_char_lm.
		order: The length of the n-grams in the language model.
	'''
	data = open(test_filename).read()
	pad = "~" * order
	test = pad + data
  
	# TODO: YOUR CODE HRE
	p=0
	for i in range(len(test)-order):
		history, char=test[i:i+order],test[i+order]
		if history in lm:
			next_char=[x for x,_ in lm[history]]
			if char in next_char:
				ix=next_char.index(char)
				p+=np.log(lm[history][ix][1])
			else:
				return np.inf
		else:
			return np.inf
	p*=(-1./len(test))
	return np.exp(p)
            
def calculate_oov_prob(data, lm):
#    lm here is unigram model
    tuples=lm['']
    vocab=set([x for x,_ in tuples])
    freq=0
    for char in data:
        if not char in vocab:
            freq+=1
    return float(freq)/float(len(data))

def calculate_prob_with_backoff(char, history, lms, lambdas):
    '''Uses interpolation to compute the probability of char given a series of 
        language models trained with different length n-grams.

    Inputs:
        char: Character to compute the probability of.
        history: A sequence of previous text.
        lms: A list of language models, outputted by calling train_char_lm.
        lambdas: A list of weights for each lambda model. These should sum to 1.
    
    Returns:
        Probability of char appearing next in the sequence.
    ''' 
    # TODO: YOUR CODE HRE
    #Assuming lms are in reverse chronological order
    
    p=0
    hist=history
    for i in range(len(lms)-1):
        if hist in lms[i]:
            tuples=list(lms[i][hist])
            next_char=[x for x,_ in tuples]
            if char in next_char:
                ix=next_char.index(char)
                p+=lambdas[i]*tuples[ix][1]
        hist=hist[1:]
                
    tuples=lms[-1]['']
    next_char=[x for x,_ in tuples]
    if char in next_char:
        ix=next_char.index(char)
        p+=lambdas[-1]*tuples[ix][1]

    return p
    pass


def set_lambdas(lms, dev_filename):
    '''Returns a list of lambda values that weight the contribution of each n-gram model

        This can either be done heuristically or by using a development set.

    Inputs:
        lms: A list of language models, outputted by calling train_char_lm.
        dev_filename: Path to a development text file to optionally use for tuning the lmabdas. 

    Returns:
        Probability of char appearing next in the sequence.
    '''
    # TODO: YOUR CODE HERE
    with open(dev_filename) as test:
        dev=test.read()
    
    oov_prob=calculate_oov_prob(dev,lms[-1])
    
    order=len(lms)
    pad='*'*order
    dev=pad+dev
    
    good_lambdas=[]
    good_p=np.inf
    iterations=1000
    while iterations:
        iterations-=1
        
        lambdas=np.random.random(order)
        lambdas=np.divide(lambdas,np.sum(lambdas))
        
        p=0
        
        for i in np.arange(len(dev)-order):
            history,char=dev[i:i+order],dev[i+order]
            cal=calculate_prob_with_backoff(char,history,lms,lambdas)
            #print (cal)
            if cal>0:
                p+=np.log(cal)
            else:
                p=np.log(oov_prob)
                
        #print (p)
        p*=-1/len(dev)
        p=np.exp(p)
        if p<good_p:
            good_p=p
            good_lambdas=lambdas
#            print (good_lambdas)
                
    return good_lambdas
    pass

def calculate_perplexity_with_backoff(lms, filename, lambdas):
#    with open(filename) as test:
    dev=codecs.open(filename, 'r', encoding='utf8', errors='replace').read()
    
    oov_prob=calculate_oov_prob(dev,lms[-1])
    
    
    order=len(lms)
    pad='*'*order
    dev=pad+dev
    
    p=0
    
    for i in np.arange(len(dev)-order):
        history,char=dev[i:i+order],dev[i+order]
        cal=calculate_prob_with_backoff(char,history,lms,lambdas)
        #print (cal)
        if cal>0:
            p+=np.log(cal)
        else:
            p=np.log(oov_prob)
    
    p*=(-1./len(dev))
    return np.exp(p)
    pass
    
if __name__ == '__main__':
  print('Training language model')
  lm = train_char_lm("jane_austen.txt", order=2)
  #print(generate_text(lm, 4,5000))
