# Dataset handler for binary classification tasks (MR, CR, SUBJ, MQPA)

import numpy as np
from numpy.random import RandomState
import os.path


def load_data(encoder, name, loc='./data/', seed=1234):
    """
    Load one of MR, CR, SUBJ or MPQA
    """
    z = {}
    if name == 'MR':
        pos, neg = load_rt(loc=loc)
    elif name == 'SUBJ':
        pos, neg = load_subj(loc=loc)
    elif name == 'CR':
        pos, neg = load_cr(loc=loc)
    elif name == 'MPQA':
        pos, neg = load_mpqa(loc=loc)

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    z['text'] = text
    z['labels'] = labels
    print 'Computing skip-thought vectors...'
    features = encoder.encode(text, verbose=False)
    return z, features


def load_rt(loc='./data/'):
    """
    Load the MR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'rt-polarity.pos'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'rt-polarity.neg'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_subj(loc='./data/'):
    """
    Load the SUBJ dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'plot.tok.gt9.5000'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'quote.tok.gt9.5000'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_cr(loc='./data/'):
    """
    Load the CR dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'custrev.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'custrev.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def load_mpqa(loc='./data/'):
    """
    Load the MPQA dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'mpqa.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'mpqa.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels


def shuffle_data(X, L, seed=1234):
    """
    Shuffle the data
    """
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return (X, L)    




