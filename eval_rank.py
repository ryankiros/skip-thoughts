'''
Evaluation code for image-sentence ranking
'''
import numpy as np

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import os
import time

from scipy import optimize, stats
from scipy.linalg import norm
from collections import OrderedDict
from sklearn.cross_validation import KFold
from numpy.random import RandomState

import warnings


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# all parameters
def init_params(options):
    """
    Initalize all model parameters here
    """
    params = OrderedDict()

    # Image embedding, sentence embedding
    params = get_layer('ff')[0](options, params, prefix='ff_im', nin=options['dim_im'], nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_s', nin=options['dim_s'], nout=options['dim'])

    return params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]
    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer')}

def get_layer(name):
    """
    Part of the reason the init is very slow is because,
    the layer's constructor is called even when it isn't needed
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def norm_weight(nin,nout=None):
    """
    Weight initialization
    """
    if nout == None:
        nout = nin
    else:
        r = numpy.sqrt( 2. / nin)
        W = numpy.random.rand(nin, nout) * 2 * r - r
    return W.astype('float32')

def linear(x):
    return x

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# L2norm, row-wise
def l2norm(X):
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X

# build a training model
def build_model(tparams, options):
    """
    Construct computation graph for the whole model
    """
    # inputs (image, sentence, contrast images, constrast sentences)
    im = tensor.matrix('im', dtype='float32')
    s = tensor.matrix('s', dtype='float32')
    cim = tensor.matrix('cim', dtype='float32')
    cs = tensor.matrix('cs', dtype='float32')

    # image embedding
    lim = get_layer('ff')[1](tparams, im, options, prefix='ff_im', activ='linear')
    lcim = get_layer('ff')[1](tparams, cim, options, prefix='ff_im', activ='linear')

    # sentence embedding
    ls = get_layer('ff')[1](tparams, s, options, prefix='ff_s', activ='linear')
    lcs = get_layer('ff')[1](tparams, cs, options, prefix='ff_s', activ='linear')

    # L2 norm for sentences
    ls = l2norm(ls)
    lcs = l2norm(lcs)

    # Tile by number of contrast terms
    lim = tensor.tile(lim, (options['ncon'], 1))
    ls = tensor.tile(ls, (options['ncon'], 1))

    # pairwise ranking loss
    cost_im = options['margin'] - (lim * ls).sum(axis=1) + (lim * lcs).sum(axis=1)
    cost_im = cost_im * (cost_im > 0.)
    cost_im = cost_im.sum(0)

    cost_s = options['margin'] - (ls * lim).sum(axis=1) + (ls * lcim).sum(axis=1)
    cost_s = cost_s * (cost_s > 0.)
    cost_s = cost_s.sum(0)

    cost = cost_im + cost_s
    return [im, s, cim, cs], cost

# build an encoder
def build_encoder(tparams, options):
    """
    Construct encoder
    """
    # inputs (image, sentence)
    im = tensor.matrix('im', dtype='float32')
    s = tensor.matrix('s', dtype='float32')

    # embeddings
    eim = get_layer('ff')[1](tparams, im, options, prefix='ff_im', activ='linear')
    es = get_layer('ff')[1](tparams, s, options, prefix='ff_s', activ='linear')

    # L2 norm of rows
    lim = l2norm(eim)
    ls = l2norm(es)

    return [im, s], lim, ls

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * numpy.float32(0.))
        v = theano.shared(p.get_value() * numpy.float32(0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

# things to avoid doing 
def validate_options(options):

    if options['dim'] > options['dim_im']:
        warnings.warn('dim should not be bigger than image dimension')
    if options['dim'] > options['dim_s']:
        warnings.warn('dim should not be bigger than sentence dimension')
    if options['margin'] > 1:
        warnings.warn('margin should not be bigger than 1')
    return options

# Load a saved model and evaluate the results
def evaluate(X, saveto, evaluate=False, out=False):
    print "Loading model..."
    with open('%s.pkl'%saveto, 'rb') as f:
        model_options = pkl.load(f)

    params = init_params(model_options)
    params = load_params(saveto, params)
    tparams = init_tparams(params)

    print 'Building encoder'
    inps_e, lim, ls = build_encoder(tparams, model_options)
    f_emb = theano.function(inps_e, [lim, ls], profile=False)

    print 'Compute embeddings...'
    lim, ls = f_emb(X[1], X[2])

    if evaluate:
        (r1, r5, r10, medr) = i2t(lim, ls)
        print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
        (r1i, r5i, r10i, medri) = t2i(lim, ls)
        print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)
    if out:
        return lim, ls

# trainer
def trainer(train, dev, # training and development tuples
            dim=1000, # embedding dimensionality
            dim_im=4096, # image dimensionality
            dim_s=4800, # sentence dimensionality
            margin=0.2, # margin for pairwise ranking
            ncon=50, # number of contrastive terms
            max_epochs=15,
            lrate=0.01, # not needed with Adam
            dispFreq=10,
            optimizer='adam',
            batch_size = 100,
            valid_batch_size = 100,
            saveto='/ais/gobi3/u/rkiros/ssg/models/cocorank1000_combine.npz',
            validFreq=500,
            saveFreq=500,
            reload_=False):

    # Model options
    model_options = {}
    model_options['dim'] = dim
    model_options['dim_im'] = dim_im
    model_options['dim_s'] = dim_s
    model_options['margin'] = margin
    model_options['ncon'] = ncon
    model_options['max_epochs'] = max_epochs
    model_options['lrate'] = lrate
    model_options['dispFreq'] = dispFreq
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['valid_batch_size'] = valid_batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['saveFreq'] = saveFreq
    model_options['reload_'] = reload_

    model_options = validate_options(model_options)
    print model_options

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    inps, cost = build_model(tparams, model_options)

    print 'Building encoder'
    inps_e, lim, ls = build_encoder(tparams, model_options)

    print 'Building functions'
    f_cost = theano.function(inps, -cost, profile=False)
    f_emb = theano.function(inps_e, [lim, ls], profile=False)

    # gradient computation
    print 'Computing gradients'
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    uidx = 0
    estop = False
    start = 1234
    seed = 1234
    inds = numpy.arange(len(train[0]))
    numbatches = len(inds) / batch_size
    curr = 0
    counter = 0
    target=None
    history_errs = []

    # Main loop
    for eidx in range(max_epochs):
        tic = time.time()
        prng = RandomState(seed - eidx - 1)
        prng.shuffle(inds)

        for minibatch in range(numbatches):

            uidx += 1
            conprng_im = RandomState(seed + uidx + 1)
            conprng_s = RandomState(2*seed + uidx + 1)

            im = train[1][inds[minibatch::numbatches]]
            s = train[2][inds[minibatch::numbatches]]

            cinds_im = conprng_im.random_integers(low=0, high=len(train[0])-1, size=ncon * len(im))
            cinds_s = conprng_s.random_integers(low=0, high=len(train[0])-1, size=ncon * len(s))
            cim = train[1][cinds_im]
            cs = train[2][cinds_s]

            ud_start = time.time()
            cost = f_grad_shared(im, s, cim, cs)
            f_update(lrate)
            ud_duration = time.time() - ud_start

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud_duration

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing ranks...'
                lim, ls = f_emb(dev[1], dev[2])
                (r1, r5, r10, medr) = i2t(lim, ls)
                print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
                (r1i, r5i, r10i, medri) = t2i(lim, ls)
                print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

                currscore = r1 + r5 + r10 + r1i + r5i + r10i
                if currscore > curr:
                    curr = currscore

                    # Save model
                    print 'Saving...',
                    params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'


def i2t(images, captions, npts=None):
    """
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.shape[0] / 5
    index_list = []

    # Project captions
    for i in range(len(captions)):
        captions[i] /= norm(captions[i])

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])
        im /= norm(im)

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5*index, 5*index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i(images, captions, npts=None):
    """
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    # Project images
    for i in range(len(ims)):
        ims[i] /= norm(ims[i])

    # Project captions
    for i in range(len(captions)):
        captions[i] /= norm(captions[i])

    ranks = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]

        # Compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


