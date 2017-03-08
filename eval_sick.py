'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''
import numpy as np
import os.path
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam


def evaluate(encoder, seed=1234, evaltest=False, loc='./data/'):
    """
    Run experiment
    """
    print 'Preparing data...'
    train, dev, test, scores = load_data(loc)
    train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)
    
    print 'Computing training skipthoughts...'
    trainA = encoder.encode(train[0], verbose=False, use_eos=True)
    trainB = encoder.encode(train[1], verbose=False, use_eos=True)
    
    print 'Computing development skipthoughts...'
    devA = encoder.encode(dev[0], verbose=False, use_eos=True)
    devB = encoder.encode(dev[1], verbose=False, use_eos=True)

    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Encoding labels...'
    trainY = encode_labels(scores[0])
    devY = encode_labels(scores[1])

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1])

    if evaltest:
        print 'Computing test skipthoughts...'
        testA = encoder.encode(test[0], verbose=False, use_eos=True)
        testB = encoder.encode(test[1], verbose=False, use_eos=True)

        print 'Computing feature combinations...'
        testF = np.c_[np.abs(testA - testB), testA * testB]

        print 'Evaluating...'
        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
        pr = pearsonr(yhat, scores[2])[0]
        sr = spearmanr(yhat, scores[2])[0]
        se = mse(yhat, scores[2])
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)

        return yhat


def prepare_model(ninputs=9600, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    lrmodel = Sequential()
    lrmodel.add(Dense(input_dim=ninputs, output_dim=nclass))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel


def train_model(lrmodel, X, Y, devX, devY, devscores):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1,6)
    
    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        yhat = np.dot(lrmodel.predict_proba(devX, verbose=2), r)
        score = pearsonr(yhat, devscores)[0]
        if score > best:
            print score
            best = score
            bestlrmodel = prepare_model(ninputs=X.shape[1])
            bestlrmodel.set_weights(lrmodel.get_weights())
        else:
            done = True

    yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
    score = pearsonr(yhat, devscores)[0]
    print 'Dev Pearson: ' + str(score)
    return bestlrmodel
    

def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y


def load_data(loc='./data/'):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(os.path.join(loc, 'SICK_train.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(os.path.join(loc, 'SICK_trial.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])
    with open(os.path.join(loc, 'SICK_test_annotated.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    devS = [float(s) for s in devS[1:]]
    testS = [float(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [devA[1:], devB[1:]], [testA[1:], testB[1:]], [trainS, devS, testS]


