import time
import numpy as np

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from keras.models import Model
from keras.layers import \
    Input, Dense, Bidirectional \
    , Dropout, BatchNormalization, Conv1D, GRU \
    , GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers.embeddings import Embedding

from keras.regularizers import l1_l2


#/////////////////////////////////////////////////////////////////////////////////////


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained fastText 300-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their word_to_vec_map vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["bonjour"].shape[0]       # define dimensionality of our fastText word vectors (= 300)
    
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    emb_matrix = np.zeros( (vocab_len, emb_dim) )
    
    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[ idx, : ] = word_to_vec_map[ word ]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Making it non-trainable.
    embedding_layer = Embedding(
        input_dim = vocab_len,
        output_dim = emb_dim,
        trainable = False
    )

    # Step 4
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Our layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


#/////////////////////////////////////////////////////////////////////////////////////


def build_model( input_shape, word_to_vec_map, word_to_index
                , spatial_dropout_prop = 0.0
                , recurr_units = 1
                , recurrent_regularizer_l1_factor = 0.0
                , recurrent_regularizer_l2_factor = 0.0
                , recurrent_dropout_prop = 0.0
                , conv_units = 1
                , kernel_size_1 = 1
                , kernel_size_2 = 1
                , dense_units_1 = 1
                , dense_units_2 = 1
                , dropout_prop = 0.0
               ) :
    """
    Function creating the model's graph.
    
    Arguments:
        - input_shape     -- shape of the input, usually (max_len,)
        - word_to_vec_map -- dictionary mapping every word in a vocabulary
                             into its 50-dimensional vector representation
        - word_to_index   -- dictionary mapping from words to their indices
                             in the vocabulary

    Returns:
        - model           -- a model instance in Keras
    """

    tic = time.perf_counter()

    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32'
    # (as it contains indices, which are integers).
    sentence_indices = Input( shape = input_shape, dtype = 'int32' )

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer( word_to_vec_map, word_to_index )
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer( sentence_indices )

    x1 = SpatialDropout1D(spatial_dropout_prop)(embeddings)

    x_gru_1 = Bidirectional(GRU(int(recurr_units)
                                , recurrent_regularizer = \
                                       l1_l2(l1=recurrent_regularizer_l1_factor
                                           , l2=recurrent_regularizer_l2_factor)
                                , recurrent_dropout = recurrent_dropout_prop
                                , return_sequences = True))(x1)
    x_gru_2 = Bidirectional(GRU(int(recurr_units)
                                , recurrent_regularizer = \
                                       l1_l2(l1=recurrent_regularizer_l1_factor
                                           , l2=recurrent_regularizer_l2_factor)
                                , recurrent_dropout = recurrent_dropout_prop
                                , return_sequences = True))(x1)


    x1 = Conv1D(int(conv_units), kernel_size=int(kernel_size_1), padding='valid'
                , kernel_initializer='he_uniform')(x_gru_1)
    max_pool1_gru_1 = GlobalMaxPooling1D()(x1)

    x2 = Conv1D(int(conv_units), kernel_size=int(kernel_size_2), padding='valid'
                , kernel_initializer='he_uniform')(x_gru_1)
    max_pool2_gru_1 = GlobalMaxPooling1D()(x2)


    x1 = Conv1D(int(conv_units), kernel_size=int(kernel_size_1), padding='valid'
                , kernel_initializer='he_uniform')(x_gru_2)
    max_pool1_gru_2 = GlobalMaxPooling1D()(x1)
    
    x2 = Conv1D(int(conv_units), kernel_size=int(kernel_size_2), padding='valid'
                , kernel_initializer='he_uniform')(x_gru_2)
    max_pool2_gru_2 = GlobalMaxPooling1D()(x2)
    
    
    x = concatenate([max_pool1_gru_1, max_pool2_gru_1,
                     max_pool1_gru_2, max_pool2_gru_2])

    x = BatchNormalization()(x)
    x = Dropout(dropout_prop)(Dense(int(dense_units_1),activation='relu') (x))

    x = BatchNormalization()(x)
    x = Dropout(dropout_prop)(Dense(int(dense_units_2),activation='relu') (x))

    x = Dense(1, activation = "relu")(x) # Linear Regression output

    # Create Model instance which converts sentence_indices into x.
    model = Model( inputs = sentence_indices, outputs = x )

    toc = time.perf_counter()
    print(f"Built the model in {toc - tic:0.4f} seconds")

    return model


#/////////////////////////////////////////////////////////////////////////////////////


































