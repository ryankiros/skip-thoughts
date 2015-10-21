# decoding

This document will describe how to train decoders conditioned on skip-thought vectors. Some example tasks include:

* Decoding: Generating the sentence that the conditioned vector had encoded
* Conversation: Generating the next sentence given the encoding of the previous sentence
* Translation: Generate a French translation given the encoding of the source English sentence.

I have only tried out the first task, so YMMV on the others but in principle it should work. We assume that you have two lists of strings available: X which are the target sentences and C which are the source sentences. The model will condition on the skip-thought vectors of sentences in C to generate the sentences in X. Note that each string in X should already be tokenized (so that split() will return the desired tokens).

### Step 1: Create a dictionary

We first need to create a dictionary of words from the target sentences X. In IPython, run the following:

    import vocab
    worddict, wordcount = vocab.build_dictionary(X)

This will return 2 dictionaries. The first maps each word to an index, while the second contains the raw counts of each word. Next, save these dictionaries somewhere:

    vocab.save_dictionary(worddict, wordcount, loc)
    
Where 'loc' is a specified path where you want to save the dictionaries.

### Step 2: Setting the hyperparameters

Open train.py with your favourite editor. The trainer functions contains a number of available options. We will step through each of these below:

* dimctx: the context vector dimensionality. Set =4800 for the model on the front page
* dim_word: the dimensionality of the RNN word embeddings
* dim: the size of the hidden state
* decoder: the type of decoder function. Only supports 'gru' at the moment
* doutput: whether to use a deep output layer
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
* embeddings: path to dictionary of pre-trained wordvecs (keys are words, values are vectors). Otherwise None
* saveFreq: save the model after this many weight updates
* sampleFreq: how often to show samples from the model
* reload_: whether to reload a previously saved model

At the moment, only 1 recurrent layer is supported. Additional functionality may be added in the future.

### Step 3: Load a pre-trained skip-thoughts model

As an example, follow the instructions on the front page to load a pre-trained model. In homogeneous_data.py, specify the path to skipthoughts.py from the main page.

### Step 4: Launch the training

Once the above settings are set as desired, we can start training a model. This can be done by running

    import train
    train.trainer(X, C, skmodel)

Where skmodel is the skip-thoughts model loaded from Step 3. As training progresses the model will periodically generate samples and compare them to the ground truth. For the decoding task, you might start seeing results like this:

    Truth  0 :  UNK in hand , I opened my door .
    Sample ( 0 )  0 :  Saber , I opened my door in .    
    Truth  1 :  Holly thanked Thomas with a smile .     
    Sample ( 0 )  1 :  Amber thanked Adam with a smile .   
    Truth  2 :  I could n't look at him . Not now .          
    Sample ( 0 )  2 :  Too could n't look at him . Not now .     
    Truth  3 :  `` And is it all about the pay ? ''          
    Sample ( 0 )  3 :  `` And is it all about the pay ? ''        
    Truth  4 :  `` What do we do now ? '' I asked .             
    Sample ( 0 )  4 :  `` What do we do now ? '' I asked .      
    Truth  5 :  `` It was n't a problem at all . ''            
    Sample ( 0 )  5 :  It was n't a problem at all . ''     
    Truth  6 :  Because this is where she belongs .     
    Sample ( 0 )  6 :  At this where she belongs .      
    Truth  7 :  Nowhere to be found , I confirmed .     
    Sample ( 0 )  7 :  Much to be found , correct .     
    Truth  8 :  But in the end , he 'd lost Henry .    
    Sample ( 0 )  8 :  Regardless in the end , he 'd lost himself .  
    Truth  9 :  `` I 'm not sorry , '' Vance said .         
    Sample ( 0 )  9 :  `` I 'm not sorry , '' Vance said .
    
At the beginning of training, the samples will look horrible. As training continues, the model will get better at trying to decode the ground truth, as shown above.

### Step 5: Loading saved models

In tools.py is a function for loading saved models. Open tools.py with your favourite editor and specify path_to_model and path_to_dictionary. Then run the following:

    import tools
    dec = tools.load_model()

The output will be a dictionary with all the components necessary to generate new text.

### Step 6: Generating text

In tools.py is a function called run_sampler which can be used to generate new text conditioned on a skip-thought vector. For example, suppose that vec is a vector encoding a sentence. We can then generate text by running

    text = tools.run_sampler(dec, vec, beam_width=1, stochastic=False, use_unk=False)
  
This will generate a sentence, conditioned on vec, using greedy decoding. If stochastic=True, it will generate a sentence by randomly sampling from the predicted distributions. If use_unk=False, the unknown token (UNK) will not be included in the vocabulary. If instead of greedy decoding, you can specify a beam width. In this case, it will output the top-K sentences for a beam width of size K.

### Training advice

I included a theano function f_log_probs in train.py which can be used for monitoring the cost on held-out data. On BookCorpus, one pass through the dataset (70 million sentences) should be good enough for very accurate decoding.

In layers.py, you can create additional types of layers to replace gru. It is just a matter of following the template of the existing layers.

Consider initializing with pre-trained word vectors. This helps get training off the ground faster.

In theory you can also backprop through the skip-thoughts encoder. The code currently doesn't support this though.

## Acknowledgements

This code was built off of [arctic-captions](https://github.com/kelvinxu/arctic-captions) and Kyunghyun Cho's [dl4mt-material](https://github.com/kyunghyuncho/dl4mt-material). A big thanks to all those who contributed to these projects.
