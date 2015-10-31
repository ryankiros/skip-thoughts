# training

This document will describe how to train new models from scratch.

## Getting started

NOTE: Make sure you have 'floatX=float32' set in your Theano flags, otherwise you may encounter a TypeError.

Suppose that you have a list of strings available for training, where the contents of the entries are contiguous (so the (i+1)th entry is the sentence that follows the i-th entry. As an example, you can download our [BookCorpus](http://www.cs.toronto.edu/~mbweb/) dataset, which was used for training the models available on the main page. Lets call this list X. Note that each string should already be tokenized (so that split() will return the desired tokens).

### Step 1: Create a dictionary

We first need to create a dictionary of words from the corpus. In IPython, run the following:

    import vocab
    worddict, wordcount = vocab.build_dictionary(X)

This will return 2 dictionaries. The first maps each word to an index, while the second contains the raw counts of each word. Next, save these dictionaries somewhere:

    vocab.save_dictionary(worddict, wordcount, loc)
    
Where 'loc' is a specified path where you want to save the dictionaries.

### Step 2: Setting the hyperparameters

Open train.py with your favourite editor. The trainer functions contains a number of available options. We will step through each of these below:

* dim_word: the dimensionality of the RNN word embeddings
* dim: the size of the hidden state
* encoder: the type of encoder function. Only supports 'gru' at the moment
* decoder: the type of decoder function. Only supports 'gru' at the moment
* max_epochs: the total number of training epochs
* displayFreq: display progress after this many weight updates
* decay_c: weight decay hyperparameter
* grad_clip: gradient clipping hyperparamter
* n_words: the size of the decoder vocabulary
* maxlen_w: the max number of words per sentence. Sentences longer than this will be ignored
* optimizer: the optimization algorithm to use. Only supports 'adam' at the moment
* batch_size: size of each training minibatch (roughly)
* saveto: a path where the model will be periodically saved
* dictionary: where the dictionary is. Set this to where you saved in Step 1
* saveFreq: save the model after this many weight updates
* reload_: whether to reload a previously saved model

At the moment, only 1 layer models are supported. Additional functionality may be added in the future.

### Step 3: Launch the training

Once the above settings are set as desired, we can start training a model. This can be done by running

    import train
    train.trainer(X)

It will take a few minutes to load the dictionary and compile the model. After this is done, it should start printing out progress, like this:

    Epoch  0 Update  1 Cost  5767.91308594 UD  2.27778100967
    Epoch  0 Update  2 Cost  4087.91357422 UD  2.10255002975
    Epoch  0 Update  3 Cost  5373.07714844 UD  2.42809081078
    
The Cost is the total sum of the log probabilities across each batch, timestep and forward/backward decoder. The last number shows how long it took to do a single iteration (forward pass, backward pass and weight update). Note that the Cost will fluxuate a lot, since it is not normalized by the sentence length.

Training works by grouping together examples of the same length for the encoder. Thus, the decoder sentences all have different lengths. To accommodate this, we use a masking parameter which can copy over the state of shorter sentences in the decoder. This mask is also used when computing the loss to ignore unwanted timesteps.

NOTE: training takes a long time! Please be patient. On BookCorpus, you should start getting good sentence vectors after about 3-4 days of training on a modern GPU (the results on the tasks used in the paper should be in the same ballpark as the model on the front page, but slightly worse). The pre-trained models on the front page were trained for 2 weeks.
  
### Step 4: Loading saved models

In tools.py is a function for loading saved models. Open tools.py with your favourite editor and specify path_to_model, path_to_dictionary and path_to_word2vec. Word2vec is used for doing vocabulary expansion (see the paper for more details). We used the publicly available pre-trained Google News vectors from [here](https://code.google.com/p/word2vec/).

Once these are specified, run the following:

    import tools
    embed_map = tools.load_googlenews_vectors()
    model = tools.load_model(embed_map)

This will return a dictionary containing all the functions necessary for encoding new sentences. Note that loading will take a few minutes, due to the vocabulary expansion step. The output is largely similiar to the output of skipthoughts.load_model() on the main page.

### Step 5: Encoding new sentences

Once the model is loaded, encoding new sentences into vectors is easy. Just run

    vectors = tools.encode(model, X)
  
Where X is a list of strings to encode. This functionality is near equivalent to skipthoughts.encode on the main page.

### Training advice

In my experience, the bigger the state and the longer the training, the better the vectors you get. Out of the other hyperparameters, grad_clip is also worth tuning if possible. This code does not do any early stopping or validation (since this was not necessary for us). I included a theano function f_log_probs in train.py which can be used for monitoring the cost on held-out data, if this is necessary for you.

In layers.py, you can create additional types of layers to replace gru. It is just a matter of following the template of the existing layers.

We are working on faster versions of skip-thoughts which can be trained in hours (instead of days!). These will eventually make their way here.

## Acknowledgements

This code was built off of [arctic-captions](https://github.com/kelvinxu/arctic-captions) and Kyunghyun Cho's [dl4mt-material](https://github.com/kyunghyuncho/dl4mt-material). A big thanks to all those who contributed to these projects.
