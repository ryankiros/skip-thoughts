# Evaluation for MSRP

import numpy as np

from collections import defaultdict
from nltk.tokenize import word_tokenize
from numpy.random import RandomState
import os.path
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1


def evaluate(encoder, k=10, seed=1234, evalcv=True, evaltest=False, use_feats=True, loc='./data/'):
    """
    Run experiment
    k: number of CV folds
    test: whether to evaluate on test set
    """
    print 'Preparing data...'
    traintext, testtext, labels = load_data(loc)

    print 'Computing training skipthoughts...'
    trainA = encoder.encode(traintext[0], verbose=False)
    trainB = encoder.encode(traintext[1], verbose=False)

    if evalcv:
        print 'Running cross-validation...'
        C = eval_kfold(trainA, trainB, traintext, labels[0], shuffle=True, k=10, seed=1234, use_feats=use_feats)

    if evaltest:
        if not evalcv:
            C = 4    # Best parameter found from CV (combine-skip with use_feats=True)

        print 'Computing testing skipthoughts...'
        testA = encoder.encode(testtext[0], verbose=False)
        testB = encoder.encode(testtext[1], verbose=False)

        if use_feats:
            train_features = np.c_[np.abs(trainA - trainB), trainA * trainB, feats(traintext[0], traintext[1])]
            test_features = np.c_[np.abs(testA - testB), testA * testB, feats(testtext[0], testtext[1])]
        else:
            train_features = np.c_[np.abs(trainA - trainB), trainA * trainB]
            test_features = np.c_[np.abs(testA - testB), testA * testB]

        print 'Evaluating...'
        clf = LogisticRegression(C=C)
        clf.fit(train_features, labels[0])
        yhat = clf.predict(test_features)
        print 'Test accuracy: ' + str(clf.score(test_features, labels[1]))
        print 'Test F1: ' + str(f1(labels[1], yhat))


def load_data(loc='./data/'):
    """
    Load MSRP dataset
    """
    trainloc = os.path.join(loc, 'msr_paraphrase_train.txt')
    testloc = os.path.join(loc, 'msr_paraphrase_test.txt')

    trainA, trainB, testA, testB = [],[],[],[]
    trainS, devS, testS = [],[],[]

    f = open(trainloc, 'rb')
    for line in f:
        text = line.strip().split('\t')
        trainA.append(' '.join(word_tokenize(text[3])))
        trainB.append(' '.join(word_tokenize(text[4])))
        trainS.append(text[0])
    f.close()
    f = open(testloc, 'rb')
    for line in f:
        text = line.strip().split('\t')
        testA.append(' '.join(word_tokenize(text[3])))
        testB.append(' '.join(word_tokenize(text[4])))
        testS.append(text[0])
    f.close()

    trainS = [int(s) for s in trainS[1:]]
    testS = [int(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [testA[1:], testB[1:]], [trainS, testS]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def feats(A, B):
    """
    Compute additional features (similar to Socher et al.)
    These alone should give the same result from their paper (~73.2 Acc)
    """
    tA = [t.split() for t in A]
    tB = [t.split() for t in B]
    
    nA = [[w for w in t if is_number(w)] for t in tA]
    nB = [[w for w in t if is_number(w)] for t in tB]

    features = np.zeros((len(A), 6))

    # n1
    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]):
            features[i,0] = 1.

    # n2
    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]) and len(nA[i]) > 0:
            features[i,1] = 1.

    # n3
    for i in range(len(A)):
        if set(nA[i]) <= set(nB[i]) or set(nB[i]) <= set(nA[i]): 
            features[i,2] = 1.

    # n4
    for i in range(len(A)):
        features[i,3] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tA[i]))

    # n5
    for i in range(len(A)):
        features[i,4] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tB[i]))

    # n6
    for i in range(len(A)):
        features[i,5] = 0.5 * ((1.0*len(tA[i]) / len(tB[i])) + (1.0*len(tB[i]) / len(tA[i])))

    return features


def eval_kfold(A, B, train, labels, shuffle=True, k=10, seed=1234, use_feats=False):
    """
    Perform k-fold cross validation
    """
    # features
    labels = np.array(labels)
    if use_feats:
        features = np.c_[np.abs(A - B), A * B, feats(train[0], train[1])]
    else:
        features = np.c_[np.abs(A - B), A * B]

    scan = [2**t for t in range(0,9,1)]
    npts = len(features)
    kf = KFold(npts, n_folds=k, shuffle=shuffle, random_state=seed)
    scores = []

    for s in scan:

        scanscores = []

        for train, test in kf:

            # Split data
            X_train = features[train]
            y_train = labels[train]
            X_test = features[test]
            y_test = labels[test]

            # Train classifier
            clf = LogisticRegression(C=s)
            clf.fit(X_train, y_train)
            yhat = clf.predict(X_test)
            fscore = f1(y_test, yhat)
            scanscores.append(fscore)
            print (s, fscore)

        # Append mean score
        scores.append(np.mean(scanscores))
        print scores

    # Get the index of the best score
    s_ind = np.argmax(scores)
    s = scan[s_ind]
    print scores
    print s
    return s


