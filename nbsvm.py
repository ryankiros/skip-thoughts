# Naive-Bayes features
# Derived from https://github.com/mesnilgr/nbsvm

import os
import pdb
import numpy as np
from collections import Counter
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix


def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens


def build_dict(X, grams):
    dic = Counter()
    for sentence in X:
        dic.update(tokenize(sentence, grams))
    return dic


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r


def process_text(text, dic, r, grams):
    """
    Return sparse feature matrix
    """
    X = lil_matrix((len(text), len(dic)))
    for i, l in enumerate(text):
        tokens = tokenize(l, grams)
        indexes = []
        for t in tokens:
            try:
                indexes += [dic[t]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexes.sort()
        for j in indexes:
            X[i,j] = r[j]
    return csr_matrix(X)

