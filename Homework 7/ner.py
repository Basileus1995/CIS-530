import nltk
from nltk.corpus import conll2002
from nltk.stem.snowball import SpanishStemmer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB


# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, pos, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    contains_hyphen = 0
    if "-" in word:
        contains_hyphen = 1

    contains_apostrophe = 0
    if "'" in word:
        contains_apostrophe = 1

    contains_dot = 0
    if '.' in word:
        contains_dot = 1

    contains_NN = 0
    if 'NN' in pos:
        contains_NN = 1   

    stemmer = SpanishStemmer()    
    # print((nltk.pos_tag([word])[0][1]))

    o = str(o)
    features = [
        (o + 'word', word), (o + 'isupper', word.isupper()), (o + 'contains_apostrophe', contains_apostrophe), (o + 'contains_dot', contains_dot),
        (o + 'contains_NN', contains_NN), (o + 'contains_hyphen', contains_hyphen), (o + 'stem', stemmer.stem(word)), 
        (o + 'istitle', word.istitle()), (o + 'isdigit', word.isdigit()), (o + 'prefix', word[:3]), (o + 'suffix', word[-3:])
        ]
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-3,-2,-1,0,1,2,3]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos = sent[i+o][1]
            featlist = getfeats(word, pos, o)
            features.extend(featlist)
    
    return dict(features)

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    train_sents = train_sents + dev_sents
    test_sents = list(conll2002.iob_sents('esp.testb'))
    print("data loaded, ready for sentence parsing")
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    print("labels in place, let's vectorize and build X_train")        
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    print("Let's choose some model")

    # TODO: play with other models
    #model = Perceptron(verbose=1)
    #model = MLPClassifier(hidden_layer_sizes = 1) taking too long to run, couldn't check its performance
    #model = SGDClassifier()
    #model = PassiveAggressiveClassifier()
    #model = MultinomialNB()
    model  = LogisticRegression(solver='sag', tol=5e-3, C=1.0)
    print("Now we'll fit the model")
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    print("time for prediction!")
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("constrained_results.txt", "w") as out:
        for sent in test_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
