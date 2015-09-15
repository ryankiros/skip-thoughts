"""
A selection of functions for the decoder
Loading models, generating text
"""
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy

from utils import load_params, init_tparams
from model import init_params, build_sampler
from search import gen_sample

#-----------------------------------------------------------------------------#
# Specify model and dictionary locations here
#-----------------------------------------------------------------------------#
path_to_model = '/u/rkiros/research/semhash/models/toydec.npz'
path_to_dictionary = '/ais/gobi3/u/rkiros/flickr8k/dictionary.pkl'
#-----------------------------------------------------------------------------#

def load_model():
    """
    Load a trained model for decoding
    """
    # Load the worddict
    print 'Loading dictionary...'
    with open(path_to_dictionary, 'rb') as f:
        worddict = pkl.load(f)

    # Create inverted dictionary
    print 'Creating inverted dictionary...'
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Load model options
    print 'Loading model options...'
    with open('%s.pkl'%path_to_model, 'rb') as f:
        options = pkl.load(f)

    # Load parameters
    print 'Loading model parameters...'
    params = init_params(options)
    params = load_params(path_to_model, params)
    tparams = init_tparams(params)

    # Sampler.
    trng = RandomStreams(1234)
    f_init, f_next = build_sampler(tparams, options, trng)

    # Pack everything up
    dec = dict()
    dec['options'] = options
    dec['trng'] = trng
    dec['worddict'] = worddict
    dec['word_idict'] = word_idict
    dec['tparams'] = tparams
    dec['f_init'] = f_init
    dec['f_next'] = f_next
    return dec

def run_sampler(dec, c, beam_width=1, stochastic=False, use_unk=False):
    """
    Generate text conditioned on c
    """
    sample, score = gen_sample(dec['tparams'], dec['f_init'], dec['f_next'],
                               c.reshape(1, dec['options']['dimctx']), dec['options'],
                               trng=dec['trng'], k=beam_width, maxlen=1000, stochastic=stochastic,
                               use_unk=use_unk)
    text = []
    if stochastic:
        sample = [sample]
    for c in sample:
        text.append(' '.join([dec['word_idict'][w] for w in c[:-1]]))
    return text


