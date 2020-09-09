import sys
import time

import pandas as pd

import random
import scipy.stats as stats

import matplotlib.pyplot as plt

import multiprocessing
import dask.dataframe as dd
from tqdm import tqdm
from dask.callbacks import Callback


#/////////////////////////////////////////////////////////////////////////////////////


def get_new_hyperparameters(models_count, verbose = 0) :
    """
    Generates a set of hyperparameter values intended to be used in
    a value search hyperparameters optimization procedure when
    using a base model architecture as detailled in
    the '..architecture.build_model' method.
    The so-called values are sampled from either a logaritmic
    or a truncated normal distribution, as best fits.
    
    Parameters :
        - models_count (int) :
            how many sets of hyperparameters values must be sampled.
        - verbose (int) :
            0 (mute) / 1 (timing) / 2 (progressbar)

    Results :
        - pandas.DataFrame of shape (models_count, 14)
    """

    tic = time.perf_counter()
    
    if (verbose == 2) : print("get_new_hyperparameters - init : ", end='', file=sys.stderr, flush=True)


    new_hyperparameters = pd.DataFrame({
        "spatial_dropout_prop": \
            [round(.99-random_from_log(0, .99, base=2), 2) for _ in range(models_count)]

        , "recurr_units": \
            [int(random_from_log(4, 128, base=2)) for _ in range(models_count)]
        , "recurrent_regularizer_l1_factor": \
            [round(.05+0.005-random_from_log(0.005, .05, base=2), 5) for _ in range(models_count)]
            # higher than 0.005, maxed at 0.05, most likely 0.005, monotonic (decreasing)
            # & rather flat probability distribution
        , "recurrent_regularizer_l2_factor": \
            [round(.05+0.005-random_from_log(0.005, .05, base=2), 5) for _ in range(models_count)]
            # higher than 0.005, maxed at 0.05, most likely 0.005, monotonic (decreasing)
            # & rather flat probability distribution
        , "recurrent_dropout_prop": \
            [ round(x, 5) for x in get_truncnorm_sample(
                lower = 0, upper = 0.1, mu = .0005, sigma = .01, sample_size = models_count) ]
            # positive, maxed at 0.1, most likely 0.005, flat 'truncated normal' probability distribution

        , "conv_units": \
            [int(random_from_log(5, 32, base=2)) for _ in range(models_count)]
        , "kernel_size_1": range(models_count) # placeholder
        , "kernel_size_2": range(models_count) # placeholder

        , "dense_units_1": range(models_count) # placeholder
        , "dense_units_2": range(models_count) # placeholder
        , "dropout_prop": \
            [round(random_from_log(0, .8, base=2), 2) for _ in range(models_count)]
            # positive, maxed at 0.8, most likely 0.8, monotonic (decreasing)
            # & rather flat probability distribution

        , "lr" : \
            [round(x, 4) for x in get_truncnorm_sample(
                lower = .0003, upper = .04, mu = .03, sigma = .015, sample_size = models_count)]
        , "lr_decay_rate" : \
            [round(.8+0.05-random_from_log(0.05, .8, base=2), 2) for _ in range(models_count)]
            # higher than 0.05, maxed at 0.8, most likely 0.05, monotonic (decreasing)
            # & rather flat probability distribution
        , "lr_decay_step" : \
            [ int(x) for x in get_truncnorm_sample(lower = 10, upper = 60, mu = 30, sigma = 15
                                                   , sample_size = models_count) ]
    })

    # use dask partitions to compute the remaining (it's fairly slow otherwise)
    # start by getting the number of steps for the progress bar
    # (we need the pandas DataFrame [with its actual size & index]) =>
    npartitions = min(models_count, 4*multiprocessing.cpu_count()-1)
    ddata = dd.from_pandas(new_hyperparameters, npartitions=npartitions)
    npartitions = ddata.npartitions # since, per the doc, "the output may have fewer partitions than requested."

    toc = time.perf_counter()
    if (verbose == 2) : print(f"{toc - tic:0.4f} seconds", end='\n', file=sys.stderr, flush=True)

    if (verbose == 2) : print("get_new_hyperparameters - Step II/II :", end='\n', file=sys.stderr, flush=True)
    progress_steps = 3*npartitions + 1
    with tqdm(disable = (verbose != 2), total = progress_steps
              , bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:45}{r_bar}') as t :
        # make sure that "kernel_size_1" is not higher than "conv_units" =>
        def get_kernel_size_1(conv_units) :
            return [ int(get_truncnorm_sample(
                        lower = 2, upper = conv_units.iloc[i]
                        , mu = 3, sigma = 2, sample_size = 1)[0]
                      ) for i in range(len(conv_units)) ]
        with DaskProgressCallback(t) :
            res = ddata['conv_units'].map_partitions(get_kernel_size_1, meta=('result', int)) \
                    .compute(scheduler='processes')
        new_hyperparameters[ "kernel_size_1" ] = sum(res.values, []) # flatten partition results

        #toc = time.perf_counter()
        #print(f"step II in {toc - tic:0.4f} seconds") ; sys.stdout.flush()

        # make sure that "kernel_size_2" is not higher than "conv_units" =>
        def get_kernel_size_2(conv_units) :
            return [ int(get_truncnorm_sample(
                        lower = 2, upper = conv_units.iloc[i]
                        , mu = 3, sigma = 2, sample_size = 1)[0]
                      ) for i in range(len(conv_units)) ]

        with DaskProgressCallback(t) :
                res = ddata['conv_units'].map_partitions(get_kernel_size_2, meta=('result', int)) \
                    .compute(scheduler='processes')
        new_hyperparameters[ "kernel_size_2" ] = sum(res.values, []) # flatten partition results


        # make sure that "dense_units_1" is not higher than "4 times (4 pooling concat) conv_units" =>
        def get_dense_units_1(conv_units) :
            return [ int(get_truncnorm_sample(lower = 2, upper = 4*conv_units.iloc[i]
                                       , mu = 2*conv_units.iloc[i], sigma = 20
                                       , sample_size = 1)[0]
                      ) for i in range(len(conv_units)) ]
        
        with DaskProgressCallback(t) :
            res = ddata['conv_units'].map_partitions(get_dense_units_1, meta=('result', int)) \
                    .compute(scheduler='processes')
        new_hyperparameters[ "dense_units_1" ] = sum(res.values, []) # flatten partition results

        # make sure that "dense_units_2" is not higher than "dense_units_1" =>
        new_hyperparameters[ "dense_units_2" ] = \
             [int(random_from_log(2, new_hyperparameters["dense_units_1"][i], base=2))
              for i in range(models_count)]
        t.update()

    sys.stderr.flush()

    toc = time.perf_counter()
    if (verbose != 0) :
        print(f"Generated {models_count:,} sets of hyperparameter values " +
              f"in {toc - tic:0.4f} seconds", end='\n', file=sys.stderr, flush=True)

    return new_hyperparameters


class DaskProgressCallback(Callback):
    def __init__(self, tqdm_bar):
        super(DaskProgressCallback, self).__init__()
        self.tqdm_bar = tqdm_bar

    def _posttask(self, key, result, dsk, state, worker_id) :
        self.tqdm_bar.update()

    def _pretask(self, key, dask, state):
        """Print the key of every task as it's started"""
        #print("Computing: {0}!".format(repr(key)))
        pass


#/////////////////////////////////////////////////////////////////////////////////////


""" LOGARITHMIC DISTRIBUTION """


def random_from_log(min, max, base=10): 
    return random.uniform(min**base, max**base)**(1/base)
def random_from_exp(min, max, base=10): 
    return random.uniform(min**(1/base), max**(1/base))**base

def get_log_sample(lower = 0, upper = 1, base = 10, sample_size = 30
                   , reverse_distri = False, digits = sys.maxsize) :
    """
    Parameters :
        - lower (float):
            lower bound of the probability distribution from which the samples are drawn
        - upper (float):
            upper bound of the probability distribution from which the samples are drawn
        - base (int):
            log10 or ln or any lof base ; determines the "flatness" of
            the probability distribution from which the samples are drawn.
        - sample_size (int)
        - reverse_distri (boolean):
            whether or not the "pic" of the distribution is
            at the lower of the distribution bounds
        - digits (int):
            amount of decimals for the sample elements. '0' yields integers.
    Result :
        a list of size "sample_size"
    """
    if reverse_distri :
        return [round(upper+lower-random_from_log(lower, upper, base), digits)
                for _ in range(sample_size)]
    else :
        return [round(random_from_log(lower, upper, base), digits)
                for _ in range(sample_size)]

def log_sample_plot() :
    """
    convenience method to help developpers visualize
    a distribution generated by the 'get_log_sample' method.
    """

    x = get_log_sample(0, .004, sample_size = 3000)
    plt.hist(x, density=True, bins=30)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')

    x = get_log_sample(0, .004, 2, 3000, True)
    plt.hist(x, density=True, bins=30)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')


#/////////////////////////////////////////////////////////////////////////////////////


""" BOUNDED / TRUNCATED NORMAL DISTRIBUTION """

def get_truncnorm_sample(lower = -.2, upper = 1, mu = -.1, sigma = .1, sample_size = 30) :
    """
    """

    #instantiate an object X using the above four parameters,
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    return X.rvs(sample_size)

def truncnorm_sample_plot() :
    """
    convenience method to help developpers visualize
    a distribution generated by the 'get_truncnorm_sample' method.
    """

    lower, upper = -.2, 1
    mu, sigma = -.1, .1
    
    lower = .0003 ; upper = .04
    mu = .03 ; sigma = .015


    #generate 1000 sample data
    samples = get_truncnorm_sample(lower = lower, upper = upper, mu = mu, sigma = sigma
                                   , sample_size = 1000)


    #compute the PDF of the sample data
    pdf_probs = stats.truncnorm.pdf(samples, (lower-mu)/sigma, (upper-mu)/sigma, mu, sigma)

    #compute the CDF of the sample data
    cdf_probs = stats.truncnorm.cdf(samples, (lower-mu)/sigma, (upper-mu)/sigma, mu, sigma)

    #make a histogram for the samples
    plt.hist(samples, bins= 50,density=True,alpha=0.3,label='histogram');

    #plot the PDF curves 
    plt.plot(samples[samples.argsort()],pdf_probs[samples.argsort()],linewidth=2.3,label='PDF curve')

    #plot CDF curve        
    plt.plot(samples[samples.argsort()],cdf_probs[samples.argsort()],linewidth=2.3,label='CDF curve')


    #legend
    plt.legend(loc='best')


#/////////////////////////////////////////////////////////////////////////////////////





























