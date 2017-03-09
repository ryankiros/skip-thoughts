# Experiment scripts for binary classification benchmarks (e.g. MR, CR, MPQA, SUBJ)

import numpy as np
import sys
import nbsvm
import dataset_handler

from scipy.sparse import hstack

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold


def eval_nested_kfold(encoder, name, loc='./data/', k=10, seed=1234, use_nb=False):
    """
    Evaluate features with nested K-fold cross validation
    Outer loop: Held-out evaluation
    Inner loop: Hyperparameter tuning

    Datasets can be found at http://nlp.stanford.edu/~sidaw/home/projects:nbsvm
    Options for name are 'MR', 'CR', 'SUBJ' and 'MPQA'
    """
    # Load the dataset and extract features
    z, features = dataset_handler.load_data(encoder, name, loc=loc, seed=seed)

    scan = [2**t for t in range(0,9,1)]
    npts = len(z['text'])
    kf = KFold(npts, n_folds=k, random_state=seed)
    scores = []
    for train, test in kf:

        # Split data
        X_train = features[train]
        y_train = z['labels'][train]
        X_test = features[test]
        y_test = z['labels'][test]

        Xraw = [z['text'][i] for i in train]
        Xraw_test = [z['text'][i] for i in test]

        scanscores = []
        for s in scan:

            # Inner KFold
            innerkf = KFold(len(X_train), n_folds=k, random_state=seed+1)
            innerscores = []
            for innertrain, innertest in innerkf:
        
                # Split data
                X_innertrain = X_train[innertrain]
                y_innertrain = y_train[innertrain]
                X_innertest = X_train[innertest]
                y_innertest = y_train[innertest]

                Xraw_innertrain = [Xraw[i] for i in innertrain]
                Xraw_innertest = [Xraw[i] for i in innertest]

                # NB (if applicable)
                if use_nb:
                    NBtrain, NBtest = compute_nb(Xraw_innertrain, y_innertrain, Xraw_innertest)
                    X_innertrain = hstack((X_innertrain, NBtrain))
                    X_innertest = hstack((X_innertest, NBtest))

                # Train classifier
                clf = LogisticRegression(C=s)
                clf.fit(X_innertrain, y_innertrain)
                acc = clf.score(X_innertest, y_innertest)
                innerscores.append(acc)
                print (s, acc)

            # Append mean score
            scanscores.append(np.mean(innerscores))

        # Get the index of the best score
        s_ind = np.argmax(scanscores)
        s = scan[s_ind]
        print scanscores
        print s
 
        # NB (if applicable)
        if use_nb:
            NBtrain, NBtest = compute_nb(Xraw, y_train, Xraw_test)
            X_train = hstack((X_train, NBtrain))
            X_test = hstack((X_test, NBtest))
       
        # Train classifier
        clf = LogisticRegression(C=s)
        clf.fit(X_train, y_train)

        # Evaluate
        acc = clf.score(X_test, y_test)
        scores.append(acc)
        print scores

    return scores


def compute_nb(X, y, Z):
    """
    Compute NB features
    """
    labels = [int(t) for t in y]
    ptrain = [X[i] for i in range(len(labels)) if labels[i] == 0]
    ntrain = [X[i] for i in range(len(labels)) if labels[i] == 1]
    poscounts = nbsvm.build_dict(ptrain, [1,2])
    negcounts = nbsvm.build_dict(ntrain, [1,2])
    dic, r = nbsvm.compute_ratio(poscounts, negcounts)
    trainX = nbsvm.process_text(X, dic, r, [1,2])
    devX = nbsvm.process_text(Z, dic, r, [1,2])
    return trainX, devX



