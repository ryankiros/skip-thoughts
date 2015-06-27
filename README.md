# skip-thoughts

Sent2Vec encoder from the paper [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726). The training code and data will be released at a later date.

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [NLTK](http://www.nltk.org/)
* [Keras](https://github.com/fchollet/keras) (for Semantic-Relatedness experiments only)

## Getting started

You will first need to download the model files and word embeddings. The embedding files (utable and btable) are quite large (>2GB) so make sure there is enough space available. The encoder vocabulary can be found in dictionary.txt.

    wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
    wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

Once these are downloaded, open skipthoughts.py and set the paths to the above files (path_to_models and path_to_tables). Now you are ready to go. Make sure to set the THEANO_FLAGS device if you want to use CPU or GPU.

Open up IPython and run the following:

    import skipthoughts
    model = skipthoughts.load_model()

Now suppose you have a list of sentences X, where each entry is a string that you would like to encode. To get vectors, just run the following:

    vectors = skipthoughts.encode(X)

vectors is a numpy array with as many rows as the length of X, and each row is 4800 dimensional (combine-skip model, from the paper). The first 2400 dimensions is the uni-skip model, and the last 2400 is the bi-skip model. We highly recommend using the combine-skip vectors, as they are almost universally the best performing in the paper experiments.

As the vectors are being computed, it will print some numbers. The code works by extracting vectors in batches of sentences that have the same length - so the number corresponds to the current length being processed. If you want to turn this off, set verbose=False when calling encode.

The rest of the document will describe how to run the experiments from the paper.

## TREC Question-Type Classification

(TODO)

## Image-Sentence Ranking

(TODO)

## Semantic-Relatedness

(TODO)

## Paraphrase Detection

(TODO)

## A note about the EOS (End-of-Sentence) token

(TODO)

## Reference

If you found this code useful, please cite the following paper:

Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. **"Skip-Thought Vectors."** *arXiv preprint arXiv:1506.06726 (2015).*

    @article{kiros2015skip,
      title={Skip-Thought Vectors},
      author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
      journal={arXiv preprint arXiv:1506.06726},
      year={2015}
    }

## License

(TBD)
