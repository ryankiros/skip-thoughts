"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import defaultdict

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model, build_sampler
from vocab import load_dictionary
from search import gen_sample

# main trainer
def trainer(X, C, stmodel,
            dimctx=4800, #vector dimensionality
            dim_word=620, # word vector dimensionality
            dim=1600, # the number of GRU units
            encoder='gru',
            decoder='gru',
            doutput=False,
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=40000,
            maxlen_w=100,
            optimizer='adam',
            batch_size = 16,
            saveto='/u/rkiros/research/semhash/models/toy.npz',
            dictionary='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl',
            embeddings=None,
            saveFreq=1000,
            sampleFreq=100,
            reload_=False):

    # Model options
    model_options = {}
    model_options['dimctx'] = dimctx
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['decoder'] = decoder
    model_options['doutput'] = doutput
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['embeddings'] = embeddings
    model_options['saveFreq'] = saveFreq
    model_options['sampleFreq'] = sampleFreq
    model_options['reload_'] = reload_

    print model_options

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    # load dictionary
    print 'Loading dictionary...'
    worddict = load_dictionary(dictionary)

    # Load pre-trained embeddings, if applicable
    if embeddings != None:
        print 'Loading embeddings...'
        with open(embeddings, 'rb') as f:
            embed_map = pkl.load(f)
        dim_word = len(embed_map.values()[0])
        model_options['dim_word'] = dim_word
        preemb = norm_weight(n_words, dim_word)
        pz = defaultdict(lambda : 0)
        for w in embed_map.keys():
            pz[w] = 1
        for w in worddict.keys()[:n_words-2]:
            if pz[w] > 0:
                preemb[worddict[w]] = embed_map[w]
    else:
        preemb = None

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options, preemb=preemb)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, inps, cost = build_model(tparams, model_options)

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Done'
    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    train_iter = homogeneous_data.HomogeneousData([X,C], batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    lrate = 0.01
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for x, c in train_iter:
            n_samples += len(x)
            uidx += 1

            x, mask, ctx = homogeneous_data.prepare_data(x, c, worddict, stmodel, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, mask, ctx)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                params = unzip(tparams)
                numpy.savez(saveto, history_errs=[], **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            if numpy.mod(uidx, sampleFreq) == 0:
                x_s = x
                mask_s = mask
                ctx_s = ctx
                for jj in xrange(numpy.minimum(10, len(ctx_s))):
                    sample, score = gen_sample(tparams, f_init, f_next, ctx_s[jj].reshape(1, model_options['dimctx']), model_options,
                                               trng=trng, k=1, maxlen=100, stochastic=False, use_unk=False)
                    print 'Truth ',jj,': ',
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv],
                        else:
                            print 'UNK',
                    print
                    for kk, ss in enumerate([sample[0]]):
                        print 'Sample (', kk,') ', jj, ': ',
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in word_idict:
                                print word_idict[vv],
                            else:
                                print 'UNK',
                    print

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass


