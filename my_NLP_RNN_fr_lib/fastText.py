
import os
import time
import numpy as np

import re

from .gzip_utils import download_ungzip_progress
from .tokenizer_utils import tokenize_fr
from .tweet_utils import hastag_converter


#/////////////////////////////////////////////////////////////////////////////////////


def download_fastText_fr_300_vec():
    filefullname = os.path.join( 'data', 'fastText_french', 'cc.fr.300.vec')

    if not os.path.isfile(filefullname) :
        download_ungzip_progress(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz'
            , filefullname)

    assert (os.path.getsize(filefullname) == 4517683442), \
        'word embedding file seems to be corrupted'


#/////////////////////////////////////////////////////////////////////////////////////


both_alpha_numeric_pattern = re.compile('(?i)^(?!\d*$|[a-zÀ-ÿ]*$)[a-zÀ-ÿ\d]+$')

def read_fastText_vecs(fastText_file):
    """
    We only keep vocabulary tokens that :
        - would translate into a single token past our custom tokenizer
        - would translate into a single token past our hashtag_converter
        - are either pure text or pure number
    """
    tic = time.perf_counter()

    with open(fastText_file, 'r', encoding="utf-8", newline='\n', errors='ignore') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            if (
                len(tokenize_fr(curr_word)) == 1
                and len(both_alpha_numeric_pattern.findall(curr_word)) == 0
                and len(re.split(' ', hastag_converter(curr_word))) == 1
            ) :
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1

    toc = time.perf_counter()
    print(f"Loaded the Vocabulary in {toc - tic:0.4f} seconds [" +
          "{:,}".format(len((word_to_vec_map.keys()))) + " word embeddings of " +
          "{:,}".format( len(word_to_vec_map[list(word_to_vec_map.keys())[0]]) ) + " features" +
          "]")

    return words_to_index, index_to_words, word_to_vec_map


#/////////////////////////////////////////////////////////////////////////////////////


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to
    words in the sentences.
    The output shape should be such that it can be given to an `Embedding()` layer
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence
    in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X,
    of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros( (m, max_len) )
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split it
        # into a list of words.
        sentence_words = tokenize_fr(X[i])
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try :
                X_indices[i, j] = word_to_index[ w ]
            except KeyError as kErr :
                # case 'word is unknown' => assign the "<UKN>" token
                X_indices[i, j] = len(word_to_index)
            # Increment j to j + 1
            j = j + 1
            if j == max_len : break
    
    return X_indices









































