import os, sys
import numpy as np
import pandas as pd
import csv
import time, datetime

import dill as pickle
import statistics
import traceback

from .architecture import build_model
from .callbacks import const_lr_sheduler_func, LearningRatePrinter \
                       , InterruptOnBeacon

import tensorflow as tf

import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler \
                            , ModelCheckpoint, TensorBoard
from tqdm.keras import TqdmCallback
from keras.optimizers import RMSprop

from ..tweet_utils import TWEET_MAX_CHAR_COUNT


#/////////////////////////////////////////////////////////////////////////////////////


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#/////////////////////////////////////////////////////////////////////////////////////


def add_trained_model(
    four_datasets_tuple
    , word_to_vec_map, word_to_index

    , spatial_dropout_prop = 0.5
    , recurr_units = 128
    , recurrent_regularizer_l1_factor = 0.01
    , recurrent_regularizer_l2_factor = 0.01
    , recurrent_dropout_prop = 0.01
    , conv_units = 32
    , kernel_size_1 = 3
    , kernel_size_2 = 2
    , dense_units_1 = 64
    , dense_units_2 = 50
    , dropout_prop = 0.2

    , lr=0.001 # keras-default: 0.01
    , lr_decay_rate = 0.1, lr_decay_step = 30

    , batch_size = 512, epochs = 100

    , verbose = 1
    , timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    , is_random_set = False
    , local_path = None
    , csv_full_path = None  # csv list of pseudo-serialized models
                            # to which the new one shall be appended
) -> str :
    """
    Trains an instance of the RNN model based on the architecture
    detailed in 'my_NLP_RNN_fr_lib.model.build_model'.
    In addition :
        - Makes two local copies of that trained model instance,
          one on 'best' performance and, one at training end.
        - Also stores the training history as a pickle dump file.
        - Also appends a line to the csv file where usefull info are stored
          (hyperparameters values, performance, timestamp, local path, etc.).

    Parameters :
        - four_datasets_tuple (tuple) :
            (X_train, X_valid, y_train, y_valid) ; datasets to be used for training
        - word_to_vec_map (dict) :
            word embedding vector dictionnary ('words' as keys)
        - word_to_index (dict) :
            dictionnary of word vocabulary index ('words' as keys)

        - hyperparameter values to be used :
            o spatial_dropout_prop (float)
            o recurr_units (int)
            o recurrent_regularizer_l1_factor (float)
            o recurrent_regularizer_l2_factor (float)
            o recurrent_dropout_prop (float)
            o conv_units (int)
            o kernel_size_1 (int)
            o kernel_size_2 (int)
            o dense_units_1 (int)
            o dense_units_2 (int)
            o dropout_prop (float)
        - learning rate scheduler parameters :
            o lr (float)
            o lr_decay_rate (float)
            o lr_decay_step (int)
        - training parameters :
            o batch_size (int)
            o epochs (int)
        - verbose (int) :
            0/1
        - timestamp (string) :
            used as 'unique ID' for the model to be trained
        - is_random_set (bool) :
            flag for information - whether or not
            the set of hyperparameter values has been randomly-generated
        - local_path (string):
            parent directory to the trained model local directory.
            Parent to the folder where :
                - the 'best_model' produced by 'ModelCheckpoint' Keras callback
                - the 'last_model' saved at training end
                - the 'history_pickle' saved at training end
        - csv_full_path (string):
            full path to the csv file where the info relative to
            the newly trained model shall be appended.

    Results :
        - timestamp (str) of the trained model instance
    """

    (X_train, X_valid, y_train, y_valid) = four_datasets_tuple

    ## model architecture build ##
    my_model = build_model(
        (TWEET_MAX_CHAR_COUNT, ), word_to_vec_map, word_to_index
        , recurr_units = recurr_units
        , spatial_dropout_prop = spatial_dropout_prop
        , recurrent_regularizer_l1_factor = recurrent_regularizer_l1_factor
        , recurrent_regularizer_l2_factor = recurrent_regularizer_l2_factor
        , recurrent_dropout_prop = recurrent_dropout_prop
        , conv_units = conv_units
        , kernel_size_1 = kernel_size_1
        , kernel_size_2 = kernel_size_2
        , dense_units_1 = dense_units_1
        , dense_units_2 = dense_units_2
        , dropout_prop = dropout_prop
    )
    #my_model.summary()

    optimizer = RMSprop(learning_rate=lr)
    my_model.compile(optimizer = optimizer, loss = 'mse' #root_mean_squared_error
                     , metrics = [root_mean_squared_error])

    ## learning rate scheduler ##
    my_learning_rate_printer = None
    if verbose > 0 :
        my_learning_rate_printer = LearningRatePrinter()
    my_lr_scheduler = LearningRateScheduler(
        const_lr_sheduler_func(decay_rate = lr_decay_rate, decay_step = lr_decay_step
                               , learning_rate_printer = my_learning_rate_printer)
        , verbose=0) #verbose=1 prints at each epoch, we use a custom printer instead

    ## Tensorboard - model-specific directory ##
    log_dir = os.path.join(
    os.path.join(os.path.realpath("..\.."), "logs", "fit"
                 , timestamp))
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    #if verbose > 0 : print( log_dir )
    if verbose > 0 : print( timestamp )

    ## save best model ##
    if not local_path is None :
        mcp_save = ModelCheckpoint(
            os.path.join(local_path, 'model_' + timestamp + '_best.h5')
            , save_best_only=True, monitor='val_loss', mode='min')

    ## model training ##
    callbacks = [tensorboard_callback, my_lr_scheduler
                 , InterruptOnBeacon(), TqdmCallback(verbose=verbose)]
    if not local_path is None :
        callbacks.append(mcp_save)
    if verbose > 0 :
        callbacks.append(my_learning_rate_printer)
    history = \
        my_model.fit( X_train, y_train, batch_size = batch_size, epochs = epochs
                     , validation_data = (X_valid,y_valid)
                     , verbose=0
                     , callbacks=callbacks
                    )
    # @see https://stackoverflow.com/questions/44831317
    # (tensorboard-unble-to-get-first-event-timestamp-for-run)
    tf.summary.FileWriter(log_dir).close()

    ## trained model - store a local copy ##
    if not local_path is None :
        my_model.save(os.path.join(local_path, 'model_' + timestamp + '_last.h5'))

    ## model training history - store a local copy ##
    if not local_path is None :
        with open(os.path.join(local_path, 'train_hstory_' + timestamp + '.pickle')
                  , 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    ## pseudo-serialization to csv file ##
    if not csv_full_path is None :

        history_df = pd.DataFrame(history.history)

        best_training_step = np.argmin(np.array(history_df.val_root_mean_squared_error))
        best_val_rmse = history_df.val_root_mean_squared_error[ best_training_step ]
        val_loss = history_df.val_loss[ best_training_step ]

        val_rmse__last_10_avg = statistics.mean(history_df.val_root_mean_squared_error[-10:])
        val_loss__last_10_avg = statistics.mean(history_df.val_loss[-10:])
        train_rmse__last_10_avg = statistics.mean(history_df.root_mean_squared_error[-10:])
        train_loss__last_10_avg = statistics.mean(history_df.loss[-10:])

        fields=[timestamp, str(int(is_random_set))
                , str(len(y_train)), str(len(y_valid))
                , str(spatial_dropout_prop)

                , str(recurr_units)
                , str(recurrent_regularizer_l1_factor)
                , str(recurrent_regularizer_l2_factor)
                , str(recurrent_dropout_prop)

                , str(conv_units)
                , str(kernel_size_1)
                , str(kernel_size_2)

                , str(dense_units_1)
                , str(dense_units_2)
                , str(dropout_prop)

                , str(lr)
                , str(lr_decay_rate), str(lr_decay_step)

                , str(batch_size), str(history_df.shape[0]) # <= epochs, accounting for InterrupOnBeacon

                , str(best_training_step+1), str(best_val_rmse), str(val_loss)
                , str(val_rmse__last_10_avg), str(val_loss__last_10_avg)
                , str(train_rmse__last_10_avg), str(train_loss__last_10_avg)

                , local_path]
        try :
            with open(csv_full_path, 'at', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        except PermissionError :
            type_, value_, traceback_ = sys.exc_info()
            traceback.print_exception(type_, value_, traceback_,
                              limit=2, file=sys.stdout)

    return timestamp


#/////////////////////////////////////////////////////////////////////////////////////


def train_models(
    hyperparameters_df
    , four_datasets_tuple
    , word_to_vec_map, word_to_index

     , csv_full_path, save_root
     , batch_size = 512, epochs = 120
     , verbose = 1) :
    """
    Trains model instances with hyperparameter values as provided as input.
    The base architecture is the one constructed via the 'build_model' method.

    Parameters :
        - hyperparameters_df (pandas.dataframe):
            set of hyperparameters as generated by the 'get_new_hyperparameters'
            function. Applies to 'nrows' new models to be trained.
        - four_datasets_tuple (tuple) :
            (X_train, X_valid, y_train, y_valid) ; datasets to be used for training
        - word_to_vec_map (dict) :
            word embedding vector dictionnary ('words' as keys)
        - word_to_index (dict) :
            dictionnary of word vocabulary index ('words' as keys)

        - csv_full_path (string):
            full path to the csv file where trained models info shall be appended.
            A blank csv file is created first if non-existing.
        - save_root (string):
            parent directory to the trained models local directory.
            Parent to the folder where :
                - the 'best_model' produced by 'ModelCheckpoint' Keras callback
                - the 'last_model' saved at training end
                - the 'history_pickle' saved at training end
            are stored for each set of hyperparameter values.
        - batch_size (int):
            batch_size
        - epochs (int):
            epochs
        - verbose (int):
            0/1
    Results :
        - N/A
    """

    overall_tic = time.perf_counter()

    (X_train, X_valid, y_train, y_valid) = four_datasets_tuple

    if not os.path.isfile(csv_full_path) :
        with open(csv_full_path, 'wt', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile, delimiter = ',', fieldnames =
                    ["timestamp", "is_random_set"
                     , "training records", "validation records"
                     , "spatial_dropout_prop"

                     , "recurr_units"
                     , "recurrent_regularizer_l1_factor"
                     , "recurrent_regularizer_l2_factor"
                     , "recurrent_dropout_prop"

                     , "conv_units"
                     , "dense_units_1"
                     , "dense_units_2"

                     , "kernel_size_1"
                     , "kernel_size_2"
                     , "dropout_prop"

                     , "lr"
                     , "lr_decay_rate", "lr_decay_step"

                     , "batch_size", "epochs"

                     , "best_training_step", "best_val_rmse", "val_loss"
                     , "val_rmse__last_10_avg", "val_loss__last_10_avg"
                     , "train_rmse__last_10_avg", "train_loss__last_10_avg"

                     , "local_path"])
            writer.writeheader()

    for i, model_hyperparameters in hyperparameters_df.iterrows() :
        # LOOP OVER TRAINING CONFIGURATIONS

        model_tic = time.perf_counter()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        local_path = os.path.join(save_root, timestamp)
        os.makedirs(local_path)

        add_trained_model(
            (X_train, X_valid, y_train, y_valid)
            , word_to_vec_map, word_to_index

            , spatial_dropout_prop = model_hyperparameters['spatial_dropout_prop']

            , recurr_units = int(model_hyperparameters['recurr_units'])
            , recurrent_regularizer_l1_factor = model_hyperparameters['recurrent_regularizer_l1_factor']
            , recurrent_regularizer_l2_factor = model_hyperparameters['recurrent_regularizer_l2_factor']
            , recurrent_dropout_prop = model_hyperparameters['recurrent_dropout_prop']

            , conv_units = int(model_hyperparameters['conv_units'])
            , kernel_size_1 = int(model_hyperparameters['kernel_size_1'])
            , kernel_size_2 = int(model_hyperparameters['kernel_size_2'])

            , dense_units_1 = int(model_hyperparameters['dense_units_1'])
            , dense_units_2 = int(model_hyperparameters['dense_units_2'])
            , dropout_prop = model_hyperparameters['dropout_prop']

            , lr = model_hyperparameters['lr']
            , lr_decay_rate = model_hyperparameters['lr_decay_rate']
            , lr_decay_step = int(model_hyperparameters['lr_decay_step'])

            , batch_size = batch_size, epochs = epochs

            , verbose = verbose
            , timestamp = timestamp
            , is_random_set = model_hyperparameters['is_random_set']
            , local_path = local_path
            , csv_full_path = csv_full_path
        )
        K.clear_session()
        e = int(time.perf_counter() - overall_tic)
        e1 = int(time.perf_counter() - model_tic)
        print("model #" + str(i+1) + "/" + str(hyperparameters_df.shape[0]) + " trained " +
              f"({(e // 3600):02d}:{(e % 3600 // 60):02d}:{(e % 60):02d} " +
              f"[+{(e1 // 3600):02d}:{(e1 % 3600 // 60):02d}:{(e1 % 60):02d}])."
             )
        print()

        # BREAKS ON "TRAINING_INTERRUPT_BEACON" BEACON
        if os.path.isfile("TRAINING_INTERRUPT_BEACON"):
            sys.stderr.write('Models training interrupted - INTERRUPT beacon encountered.')
            break

    e = int(time.perf_counter() - overall_tic)
    print(f"Trained this set of #{hyperparameters_df.shape[0]:d} models " +
          f"in {(e // 3600):02d}:{(e % 3600 // 60):02d}:{(e % 60):02d}.") # extends to >24hrs


#/////////////////////////////////////////////////////////////////////////////////////













































