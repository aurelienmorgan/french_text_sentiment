import sys, os

import keras.backend as K
from keras.callbacks import Callback


#/////////////////////////////////////////////////////////////////////////////////////


class LearningRatePrinter(Callback):
    """
    Keras callback - prints the learning rate value
    when the "_lr_updated" class variable is True
    """

    def __init__( self ):
        #super( LearningRatePrinter, self ).__init__()
        super( Callback, self ).__init__()
        self.set_lr_updated(True)
    
    def set_lr_updated( self, boolean_ ) :
        self._lr_updated = boolean_

    def on_epoch_begin( self, epoch, logs={} ):
        if self._lr_updated :
            sys.stdout.write('epoch %d ; lr: %f\n' % (epoch, K.eval(self.model.optimizer.lr)))
            sys.stdout.flush()
            self.set_lr_updated(False)


#/////////////////////////////////////////////////////////////////////////////////////


def const_lr_sheduler_func(decay_rate = 0.1, decay_step = 30
                     , learning_rate_printer: LearningRatePrinter = None):
    """
    A custom learning rate scheduler function.
    Applies constant decay at constant intervals.

    Parameters:
        - decay_rate (float) :
            proportion of the old learning rate to be kept
            for the new one
        - decay_step (int) :
            number of steps between each learning rate decay
        - learning_rate_printer (LearningRatePrinter) :
            the Keras callback to be notified
            on learning rate update

    Result:
        a "schedule" function that can be passed
        to a Keras LearningRateScheduler callback
        (:class:`LearningRateScheduler <keras.callbacks.LearningRateScheduler>`).
    """
    def my_const_lr_scheduler_func(epoch, lr) :
        if epoch % decay_step == 0 and epoch:
            if not learning_rate_printer is None :
                learning_rate_printer.set_lr_updated(True)
            return lr * decay_rate
        return lr
    return my_const_lr_scheduler_func


#/////////////////////////////////////////////////////////////////////////////////////


def lr_sheduler_func(learning_rates_decays: dict
                     , learning_rate_printer: LearningRatePrinter = None):
    """
    A custom learning rate scheduler function.
    Applies varying decays at different epochs.

    Parameters:
        - learning_rates_decays (dict(int:float)) :
            dictionary of 'keys' the epochs at which
            learning-rate decay shall be applied
            and 'values' the associated rate.
        - learning_rate_printer (LearningRatePrinter) :
            the Keras callback to be notified
            on learning rate update

    Result:
        a "schedule" function that can be passed
        to a Keras LearningRateScheduler callback
        (:class:`LearningRateScheduler <keras.callbacks.LearningRateScheduler>`).
    """
    def my_lr_scheduler_func(epoch, lr) :
        if epoch and epoch in learning_rates_decays :
            if not learning_rate_printer is None :
                learning_rate_printer.set_lr_updated(True)
            return lr * learning_rates_decays[epoch]
        return lr
    return my_lr_scheduler_func


#/////////////////////////////////////////////////////////////////////////////////////


#platform independant alert/prompt
import tkinter
from tkinter import messagebox


class InterruptOnBeacon(Callback):
    """
    Extends the Keras.Callback class.
    Allows for model training to be interrupted non-abruptly
    when an "Interrupt" beacon (a file named "TRAINING_INTERRUPT_BEACON"
    in the working directory) is detected.

    The user is prompted for confirmation that the interruption shall proceed
    when such a beacon is detected.
    """

    # This code is to hide the main tkinter window
    root = tkinter.Tk()
    root.withdraw()

    def on_epoch_end(self, epoch, logs=None):
        # Looks for a file named "TRAINING_INTERRUPT_BEACON" in the working directory.
        if os.path.isfile("TRAINING_INTERRUPT_BEACON"):
            if (
                messagebox.askokcancel(
                    title = type(self).__name__
                    , message = "You are about to interrupt the model training.")
            ) :
                self.stopped_epoch = epoch
                self.model.stop_training = True


#/////////////////////////////////////////////////////////////////////////////////////






















