'''
Evaluation code for the TREC dataset
'''
import numpy as np
import os.path
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle


def evaluate(encoder, k=10, seed=1234, evalcv=True, evaltest=False, loc='./data/'):
    """
    Run experiment
    k: number of CV folds
    test: whether to evaluate on test set
    """
    print 'Preparing data...'
    traintext, testtext = load_data(loc)
    train, train_labels = prepare_data(traintext)
    test, test_labels = prepare_data(testtext)
    train_labels = prepare_labels(train_labels)
    test_labels = prepare_labels(test_labels)
    train, train_labels = shuffle(train, train_labels, random_state=seed)

    print 'Computing training skipthoughts...'
    trainF = encoder.encode(train, verbose=False, use_eos=False)
    
    if evalcv:
        print 'Running cross-validation...'
        interval = [2**t for t in range(0,9,1)]     # coarse-grained
        C = eval_kfold(trainF, train_labels, k=k, scan=interval, seed=seed)

    if evaltest:
        if not evalcv:
            C = 128     # Best parameter found from CV

        print 'Computing testing skipthoughts...'
        testF = encoder.encode(test, verbose=False, use_eos=False)

        print 'Evaluating...'
        clf = LogisticRegression(C=C)
        clf.fit(trainF, train_labels)
        yhat = clf.predict(testF)
        print 'Test accuracy: ' + str(clf.score(testF, test_labels))


def load_data(loc='./data/'):
    """
    Load the TREC question-type dataset
    """
    train, test = [], []
    with open(os.path.join(loc, 'train_5500.label'), 'rb') as f:
        for line in f:
            train.append(line.strip())
    with open(os.path.join(loc, 'TREC_10.label'), 'rb') as f:
        for line in f:
            test.append(line.strip())
    return train, test


def prepare_data(text):
    """
    Prepare data
    """
    labels = [t.split()[0] for t in text]
    labels = [l.split(':')[0] for l in labels]
    X = [t.split()[1:] for t in text]
    X = [' '.join(t) for t in X]
    return X, labels


def prepare_labels(labels):
    """
    Process labels to numerical values
    """
    d = {}
    count = 0
    setlabels = set(labels)
    for w in setlabels:
        d[w] = count
        count += 1
    idxlabels = np.array([d[w] for w in labels])
    return idxlabels


def eval_kfold(features, labels, k=10, scan=[2**t for t in range(0,9,1)], seed=1234):
    """
    Perform k-fold cross validation
    """
    npts = len(features)
    kf = KFold(npts, n_folds=k, random_state=seed)
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
            score = clf.score(X_test, y_test)
            scanscores.append(score)
            print (s, score)

        # Append mean score
        scores.append(np.mean(scanscores))
        print scores

    # Get the index of the best score
    s_ind = np.argmax(scores)
    s = scan[s_ind]
    print (s_ind, s)
    return s

