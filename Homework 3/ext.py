import os
import csv
import subprocess
import re
import random
import numpy as np
import time
#from sklearn.feature_extraction.text import TfidfTransformer

def read_in_shakespeare():
  '''Reads in the Shakespeare dataset processesit into a list of tuples.
     Also reads in the vocab and play name lists from files.

  Each tuple consists of
  tuple[0]: The name of the play
  tuple[1] A line from the play as a list of tokenized words.

  Returns:
    tuples: A list of tuples in the above format.
    document_names: A list of the plays present in the corpus.
    vocab: A list of all tokens in the vocabulary.
  '''

  tuples = []

  with open('will_play_text.csv') as f:
    csv_reader = csv.reader(f, delimiter=';')
    for row in csv_reader:
      play_name = row[1]
      line = row[5]
      line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
      line_tokens = [token.lower() for token in line_tokens]

      tuples.append((play_name, line_tokens))

  with open('vocab.txt') as f:
    vocab =  [line.strip() for line in f]

  with open('play_names.txt') as f:
    document_names =  [line.strip() for line in f]

  return tuples, document_names, vocab

def get_row_vector(matrix, row_id):
  return matrix[row_id, :]

def get_column_vector(matrix, col_id):
  return matrix[:, col_id]

def create_term_document_matrix(line_tuples, document_names, vocab):
  '''Returns a numpy array containing the term document matrix for the input lines.

  Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary

  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

  Let m = len(vocab) and n = len(document_names).

  Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of words
        and each column corresponds to a document. A_ij contains the
        frequency with which word i occurs in document j.
  '''

  vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
  docname_to_id = dict(zip(document_names, range(0, len(document_names))))

  # YOUR CODE HERE
  start_time=time.time()
#  Added np.matrix
  term_document_matrix=np.matrix(np.zeros((len(vocab),len(document_names))))
  #term_document_matrix=sc.sparse.lil_matrix((len(vocab), len(document_names)))
  for line_tuple in line_tuples:
  	doc_name=line_tuple[0]
  	words=line_tuple[1]
  	docname_index=docname_to_id[doc_name]
  	for word in words:
  		vocab_index=vocab_to_id[word]
  		term_document_matrix[vocab_index,docname_index]+=1
  print("--- %s seconds ---" % (time.time() - start_time))
  return term_document_matrix

def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    '''Returns a numpy array containing the term context matrix for the input lines.
    
    Inputs:
        line_tuples: A list of tuples, containing the name of the document and 
        a tokenized line from that document.
        vocab: A list of the tokens in the vocabulary
        
    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
        tc_matrix: A nxn numpy array where A_ij contains the frequency with which
        word j was found within context_window_size to the left or right of
        word i in any sentence in the tuples.
    '''

    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))

  # YOUR CODE HERE
    start_time=time.time()
#    term_context_matrix=sc.sparse.lil_matrix((len(vocab),len(vocab)))
#    Added np.matrix
    term_context_matrix=np.matrix(np.zeros((len(vocab),len(vocab))))
    for line_tuple in line_tuples:
        words=line_tuple[1]
        for i in range(len(words)-1):
            for j in range(i+1,min(i+context_window_size+1,len(words))):
                word1=words[i]
                word2=words[j]

                word1_index=vocab_to_id[word1]
                word2_index=vocab_to_id[word2]

                term_context_matrix[word1_index,word2_index]+=1
                term_context_matrix[word2_index,word1_index]+=1
    print("--- %s seconds ---" % (time.time() - start_time))
    return term_context_matrix

def create_PPMI_matrix(term_context_matrix):
  '''Given a term context matrix, output a PPMI matrix.
  
  See section 15.1 in the textbook.
  
  Hint: Use numpy matrix and vector operations to speed up implementation.
  
  Input:
    term_context_matrix: A nxn numpy array, where n is
        the numer of tokens in the vocab.
  
  Returns: A nxn numpy matrix, where A_ij is equal to the
     point-wise mutual information between the ith word
     and the jth word in the term_context_matrix.
  '''       
  
  # YOUR CODE HERE
  start_time=time.time()
  sum_all=np.sum(term_context_matrix)
  word_sum=np.matrix(np.sum(term_context_matrix,axis=1)+1e-6)
  print (word_sum.shape)
  ppmi_matrix=term_context_matrix*sum_all
  # ppmi_matrix=np.divide(ppmi_matrix,word_sum)
  # ppmi_matrix=np.divide(ppmi_matrix,word_sum.T)
  ppmi_matrix/=word_sum
  ppmi_matrix/=word_sum.T
  ppmi_matrix=np.log2(ppmi_matrix+1e-6)
  ppmi_matrix=np.maximum(0,ppmi_matrix)
  print ("--- %s seconds ---" % (time.time()-start_time))
  return ppmi_matrix

def create_tf_idf_matrix(term_document_matrix):
    '''Given the term document matrix, output a tf-idf weighted version.

    See section 15.2.1 in the textbook.
  
    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
        term_document_matrix: Numpy array where each column represents a document 
        and each row, the frequency of a word in that document.

    Returns:
        A numpy array with the same dimension as term_document_matrix, where
        A_ij is weighted by the inverse document frequency of document h.
    '''

    # YOUR CODE HERE
    _,num_docs=term_document_matrix.shape
    df=np.sum(term_document_matrix>0,axis=1)
#    print (df.shape)
#    Commented below line
#    df=np.matrix(df).T
    idf=np.log(float(num_docs)/df)+1.0
    tfidf=np.multiply(term_document_matrix,idf)
#    print (np.sum(tfidf-TfidfTransformer(norm=None,smooth_idf=False).fit_transform(X=term_document_matrix)))
    return tfidf

def compute_cosine_similarity(vector1, vector2):
  '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''
  
  # YOUR CODE HERE
#  print ("vector 1 shape:")
#  print (vector1.shape)
#  print ("vector 2 shape:")
#  print (vector2.shape)
  numel=np.dot(vector1,vector2.T)
#  print ("Numel")
#  print (numel)
  val1=np.sqrt(np.sum(np.square(vector1)))
  val2=np.sqrt(np.sum(np.square(vector2)))
  denom=val1*val2
#  print ("val 1:")
#  print (val1)
#  print ("val 2:")
#  print (val2)
#  print ("denom")
#  print (denom)
  return float(numel)/(float(denom)+1e-6)

def compute_jaccard_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
        vector1: A nx1 numpy array
        vector2: A nx1 numpy array

    Returns:
        A scalar similarity value.
    '''
  
    # YOUR CODE HERE
    numer=np.sum(np.minimum(vector1,vector2))
    denom=np.sum(np.maximum(vector1,vector2))
    return float(numer)/(float(denom)+1e-6)

def compute_dice_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

    Inputs:
        vector1: A nx1 numpy array
        vector2: A nx1 numpy array

    Returns:
        A scalar similarity value.
    '''

    # YOUR CODE HERE
    numer=2.0*np.sum(np.minimum(vector1,vector2))
    denom=np.sum(vector1+vector2)
    return float(numer)/(float(denom)+1e-6)

def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    ''' Ranks the similarity of all of the plays to the target play.

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

    Inputs:
        target_play_index: The integer index of the play we want to compare all others against.
        term_document_matrix: The term-document matrix as a mxn numpy array.
        similarity_fn: Function that should be used to compared vectors for two
        documents. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.

    Returns:
        A length-n list of integer indices corresponding to play names,
        ordered by decreasing similarity to the play indexed by target_play_index
    '''
  
    # YOUR CODE HERE
    _,num_docs=term_document_matrix.shape
    similarities=[]
    vector1=get_column_vector(term_document_matrix,target_play_index)
    for i in range(num_docs):
        vector2=get_column_vector(term_document_matrix,i)
        simi=similarity_fn(vector1.T,vector2.T)
        similarities.append((i,simi))
    similarities=sorted(similarities, key=lambda x: x[1],reverse=True)
    
    return [x[0] for x in similarities]

def rank_words(target_word_index, matrix, similarity_fn):
    ''' Ranks the similarity of all of the words to the target word.
    
    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.
    
    Inputs:
        target_word_index: The index of the word we want to compare all others against.
        matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
        similarity_fn: Function that should be used to compared vectors for two word
        ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
        compute_cosine_similarity.
        
    Returns:
        A length-n list of integer word indices, ordered by decreasing similarity to the 
        target word indexed by word_index
    '''
    
    # YOUR CODE HERE
    num_words,_=matrix.shape
    similarities=[]
    vector1=get_row_vector(matrix,target_word_index)
    for i in range(num_words):
        vector2=get_row_vector(matrix,i)
        simi=similarity_fn(vector1,vector2)
        similarities.append((i,simi))
    similarities=sorted(similarities,key=lambda x:x[1],reverse=True)
    return [x[0] for x in similarities]


if __name__ == '__main__':
  tuples, document_names, vocab = read_in_shakespeare()

  print('Computing term document matrix...')
  td_matrix = create_term_document_matrix(tuples, document_names, vocab)

  print('Computing tf-idf matrix...')
  tf_idf_matrix = create_tf_idf_matrix(td_matrix)

  print('Computing term context matrix...')
  tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

  print('Computing PPMI matrix...')
  PPMI_matrix = create_PPMI_matrix(tc_matrix)

  random_idx = random.randint(0, len(document_names)-1)
  #random_idx=11
  similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar plays to "%s" using %s are:' % (document_names[random_idx], sim_fn.__qualname__))
    ranks = rank_plays(random_idx, td_matrix, sim_fn)
#    ranks = rank_plays(random_idx, tf_idf_matrix, sim_fn)
    for idx in range(0, 10):
      doc_id = ranks[idx]
      print('%d: %s' % (idx+1, document_names[doc_id]))

  word = 'juliet'
  word='troubled'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on term-context frequency matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], tc_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))

  word = 'juliet'
  word='troubled'
  vocab_to_index = dict(zip(vocab, range(0, len(vocab))))
  for sim_fn in similarity_fns:
    print('\nThe 10 most similar words to "%s" using %s on PPMI matrix are:' % (word, sim_fn.__qualname__))
    ranks = rank_words(vocab_to_index[word], PPMI_matrix, sim_fn)
    for idx in range(0, 10):
      word_id = ranks[idx]
      print('%d: %s' % (idx+1, vocab[word_id]))
