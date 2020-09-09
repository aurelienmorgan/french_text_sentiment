import sys

import pandas as pd
import numpy as np

import threading

from sklearn.model_selection import ParameterGrid, ShuffleSplit, train_test_split

from joblib import parallel_backend, Parallel, delayed

from tqdm.notebook import tqdm as notebook_tqdm


#/////////////////////////////////////////////////////////////////////////////////////

import json

class xgbR_gridsearch_cv_model(object) :
    """
    This class is used as a return type by the 'xgbR_gridsearch_cv' function.

    Similar to an object of class 'sklearn.model_selection.GridSearchCV',
    it has 'best_params_' and 'best_estimator_' attributes.
    In addition to that, it also has a 'evaluations_df' attribute,
    which encompasses detailled results of 'rmse' measures performed on all
    evaluated instances of class 'xgb.XGBRegressor'.
    """

    best_params_ = None
    best_estimator_ = None

    def __init__(self, xgb_R_base_model, xgb_R_evaluations_df) :
        """
        Parameters :
            - xgb_R_base_model (xgb.XGBRegressor)
            - xgb_R_evaluations_df (pandas.DataFrame) :
                the rmse measures
                    - on each cv_fold
                    - at each n_steps
                    - of each set of hyperparameter values to be evaluated
                Has 5 columns, namely :
                    - n_estimators (int)
                    - val_rmse (float)
                    - train_rmse (float)
                    - model_params (string)
                    - model_cv_fold
        """

        self.base_model = xgb_R_base_model
        self.evaluations_df = xgb_R_evaluations_df
        self.fold_aggregated_df, self.params_ranks_df = \
            xgbR_gridsearch_cv_agg(self.evaluations_df)
        self.best_params_ = dict(
            json.loads(self.params_ranks_df[self.params_ranks_df['val_best_rmse_rank']==1
                                           ]['model_params'].iloc[0].replace("'", '"')))
        self.best_estimator_ = \
            clone(self.base_model).set_params(**self.best_params_)

    def evaluations_df(self) -> pd.DataFrame :
        return self.evaluations_df

    def params_ranks_df(self) -> pd.DataFrame :
        return self.params_ranks_df

    def fold_aggregated_df(self) -> pd.DataFrame :
        return self.fold_aggregated_df


#/////////////////////////////////////////////////////////////////////////////////////


import xgboost as xgb


#/////////////////////////////////////////////////////////////////////////////////////


import dill  # the code below will fail without this line
from sklearn import clone


class ProgressParallel(Parallel):
    """
    Extension to the 'joblib.Parallel' class,
    allowing for a tqdm progressbar followup.
    """
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with notebook_tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def parallel_delayed_indices(params_list_len, cv_folds_count):
    """
    Internal tuple generator. To be called by 'xgbR_gridsearch_cv'
    within a 'joblib.Parallel delayed' context.
    """
    for params_list_idx in range(params_list_len) :
        for cv_fold_nb in range(cv_folds_count) :
            yield (params_list_idx, cv_fold_nb)


def cv_train_params(
    xgb_R
    , xgbR_X, xgbR_y
    , params
    , cv_folds
    , cv_fold_nb
    , n_steps
    , early_stopping_rounds
    , verbose
) :
    """
    Internal function. Pickleable. To be called by 'xgbR_gridsearch_cv'
    within a 'joblib.Parallel delayed' context.
    """

    #print(xgb_R, file=sys.stdout, end = '\n', flush = True)
    xgb_R = xgb_R.set_params(**params) # overwrite already assigned hyperparameters
    #print(xgb_R, file=sys.stdout, end = '\n', flush = True)

    _params_models_df = \
        pd.DataFrame(columns=('n_estimators', 'val_rmse', 'train_rmse', 'model_params', 'model_cv_fold'))

    ix_train, ix_test = cv_folds[cv_fold_nb]

    # datasets we want to be used for performance measures
    # (and returned as part of the 'fit.evals_result' object)
    eval_s = [
        (xgbR_X.iloc[ix_train], xgbR_y[ix_train])
        , (xgbR_X.iloc[ix_test], xgbR_y[ix_test]) # <= put this last, as it's the one dataset we want "early stopping" to look into
    ]

    xgb_R_fitted = xgb_R.fit(
        xgbR_X.iloc[ix_train], xgbR_y[ix_train]
        , eval_metric='rmse'
        , eval_set=eval_s

        # Validation metric needs to improve at least once
        # in every early_stopping_rounds round(s) to continue training
        , early_stopping_rounds=early_stopping_rounds

        , verbose=0
    )

    _model_validation_df = pd.DataFrame({
        'n_estimators': range(len(xgb_R_fitted.evals_result()['validation_0']['rmse']))
        , 'train_rmse': xgb_R_fitted.evals_result()['validation_0']['rmse']
        , 'val_rmse': xgb_R_fitted.evals_result()['validation_1']['rmse']})
    _model_validation_df = \
        _model_validation_df[(_model_validation_df['n_estimators'] == 0) |
                             (_model_validation_df['n_estimators'] % n_steps == 0) |
                             (_model_validation_df['n_estimators'] ==
                                  len(xgb_R_fitted.evals_result()['validation_0']['rmse'])-1)
                            ]
    _model_validation_df['model_params'] = [repr(params)] * _model_validation_df.shape[0]
    _model_validation_df['model_cv_fold'] = [cv_fold_nb] * _model_validation_df.shape[0]
    #print(_model_validation_df, file=sys.stdout, end = '\n', flush = True)
    _params_models_df = pd.concat([_params_models_df, _model_validation_df], axis = 0, sort=False)

    if (verbose > 1) :
        score = rmse_scorer(xgb_R_fitted, xgbR_X.iloc[ix_test], xgbR_y[ix_test])
        print(str(params) + " - fold #" + str(cv_fold_nb) + " - val_rmse = " + str(round(abs(score), 5))
              , file=sys.stdout, end = '\n', flush = True)

    return _params_models_df


def xgbR_gridsearch_cv(
    xgb_R:xgb.XGBRegressor
   , xgbR_X, xgbR_y, params_grid, cv_folds_count, test_size
   , n_steps = 500
   , early_stopping_rounds = 1000
   , refit=True
   , n_jobs = 1
   , par_backend = 'loky'
   , verbose = 2
) :
    """
    since, from 'sklearn.model_selection.GridSearchCV',
    we can't link a score (an 'eval_metric')
    to the set of hyperparameters to which it pertains,
    we don't use it and rather relate to the
    usage of "ParameterGrid" generator
    to iterate over the hyperparameters combinations
    for a xgboost regressor.

    Parameters :
        - xgbR_X (pandas.DataFrame) :
            dataset features
        - xgbR_y (numpy.array) :
            dataset true label
        - params_grid (dict) :
            set of hyperparameter values the combinations of which
            are to be evaluated in term of xgboost regressor performance
        - cv_folds_count (int)
        - test_size (float) :
            0-1 proportion of the dataset to be used
            for models validation
        - n_jobs (int) :
            number of threads that can run in parrallel
            for the purpose of the herein method execution
            (note that each individual xgb regressor model
            instance is trained on one single thread).
        - n_steps (int) :
            how many n_estimator steps shall separate
            two consecutive measures of model rmse.
        - early_stopping_rounds (int) :
            how many n_estimator steps shall separate
            non-enhancing perfomrance measures on the validation
            dataset.
        - par_backend (string) :
            backend to the joblib parallel jobs.
            One of 'loky', 'multiprocessing' or 'threading'.
        - refit (boolean) :
            Refit an estimator using the best found parameters
            on the whole dataset.
            The refitted estimator is made available
            at the `best_estimator_` attribute and permits
            using `predict` directly on this instance.
        - verbose (int) :
            - 0 : no progress print
            - 1 : only progressbar displayed
            - 2 : progressbar displayed + models scores printed

    Results :
        an instance of class 'xgbR_gridsearch_cv_model'
    """

    xgbR_X.reset_index(drop=True, inplace=True)
    xgbR_y.reset_index(drop=True, inplace=True)

    cv_folds = list(
        # k-folds with 'test_size' as an accessible param
        ShuffleSplit(random_state=None, n_splits=cv_folds_count, test_size=test_size)
        .split(xgbR_X, xgbR_y)
    )
    params_list = list(ParameterGrid(params_grid))
    #print("params_list : " + str(params_list), file=sys.stdout, end = '\n', flush = True)

    par_backend = par_backend if par_backend in ['loky', 'multiprocessing', 'threading'] else 'loky'


    ################################
    ## Actual GridSearchCV BEGINS ##
    ################################
    candidates_count = len(params_list)
    if (verbose > 0) :
        print("Fitting %d folds for each of %d candidate xgboost configurations, totalling %d fits" %
              (cv_folds_count, candidates_count, cv_folds_count*candidates_count)
              , file=sys.stdout, end = '\n', flush = True
             )
    # Run xgboost (one of its scikitlearn wrappers) in parallel (with n_jobs > 1)
    # using 'joblib' on the 'threading' backend
    # (which backend is needed in order to have stdout be on the Jupyter cell
    #  and not the Terminal console).
    # @see Jupyter bug : 'https://stackoverflow.com/questions/55955330/'
    # (printed-output-not-displayed-when-using-joblib-in-jupyter-notebook)
    # REMARK : sklearn.GridSearchCV.fit uses joblib.Parallel when n_jobs>1
    with parallel_backend(par_backend) :
        result_list = ProgressParallel(n_jobs = n_jobs, use_tqdm = (verbose > 0)
                                       , total = candidates_count * cv_folds_count)(
            delayed(cv_train_params)(
                clone(xgb_R), xgbR_X, xgbR_y
                , params_list[ params_list_idx ]
                , cv_folds
                , cv_fold_nb
                , n_steps
                , early_stopping_rounds
                , verbose
            )
            for params_list_idx, cv_fold_nb in parallel_delayed_indices(candidates_count, cv_folds_count)
        )
    ################################
    ##  Actual GridSearchCV ENDS  ##
    ################################


    xgbR_gridCv = xgbR_gridsearch_cv_model(xgb_R, pd.concat(result_list).reset_index(drop=True))


    if refit :
        #print('xgbR_X : ' + str(xgbR_X.shape))
        #print('xgbR_y : ' + str(xgbR_y.shape))
        xgbR_X_train, xgbR_X_valid, xgbR_y_train, xgbR_y_valid = \
            train_test_split(xgbR_X, xgbR_y, test_size=test_size)
        eval_s = [
            (xgbR_X_train, xgbR_y_train)
            , (xgbR_X_valid, xgbR_y_valid)
        ]

        xgbR_gridCv.best_estimator_.set_params(**{'n_jobs': n_jobs})
        if (verbose > 1) :
            print("best_params_ : " + str(xgbR_gridCv.best_params_), file=sys.stdout, end = '\n', flush = True)
        #print(xgbR_gridCv.best_estimator_).

        if not early_stopping_rounds is None :
            # If 'early stopping' was used during GridSearch CrossValidation,
            # Then retrieve (for the aggregation of models with 'best_params_')
            # the optimal 'n_estimators' value and use that value
            # as the 'n_estimators' for the 'reffitted' model =>
            best_params_val_rmse = xgbR_gridCv.fold_aggregated_df[
                xgbR_gridCv.fold_aggregated_df['model_params'] == str(xgbR_gridCv.best_params_)
            ]
            best_n_estimator_idx = np.argmin(np.array(
                best_params_val_rmse['val_rmse_mean']
            ))
            n_estimators = \
                best_params_val_rmse['n_estimators'].iloc[best_n_estimator_idx]
            xgbR_gridCv.best_estimator_.set_params(**{'n_estimators': n_estimators})

        if (verbose > 0) :
            print("refitting `best_estimator_` on the whole dataset..", file=sys.stdout, end = '', flush = True)

        xgbR_gridCv.best_estimator_.fit(
            xgbR_X_train, xgbR_y_train
            , eval_metric='rmse'
            , eval_set=eval_s
            , early_stopping_rounds=early_stopping_rounds
            , verbose=0
        )

        if (verbose > 0) :
            score = rmse_scorer(xgbR_gridCv.best_estimator_, xgbR_X_valid, xgbR_y_valid)
            print(" done ; val_rmse = %f." % round(abs(score), 5), file=sys.stdout, end = '\n', flush = True)

    return xgbR_gridCv


#/////////////////////////////////////////////////////////////////////////////////////


from sklearn.metrics import make_scorer, mean_squared_error
import math


#/////////////////////////////////////////////////////////////////////////////////////


def rmse_function(y_pred, y_true) :
    rmse = math.sqrt(mean_squared_error(y_pred, y_true))
    #print('RMSE: %2.3f' % rmse, flush=True)
    return rmse

def scorer_rmse_function(y_true, y_pred) :
    ###############################################################
    ## !!!! BEWARE any scorer has inversed inputs                ##
    # in 'sklearn.model_selection.GridSearchCV.scoring' function ##
    ## (y_true, y_pred), NOT (y_pred, y_true) !!!!               ##
    ###############################################################
    # even if for 'rmse' it makes no difference, being aware
    # of that fact may save some serious headaches
    # while trying to debug GridSearchCV models performance..
    return rmse_function(y_pred, y_true)

rmse_scorer = make_scorer(scorer_rmse_function, greater_is_better=False)


#/////////////////////////////////////////////////////////////////////////////////////


import multiprocessing
# Run xgboost (one of its scikitlearn wrappers) in parallel (with n_jobs > 1)
# on any 'joblib' backend other than 'threading' first
# otherwise crashes the parent Jupyter kernel when switching to that backend
# (which backend is needed in order to have stdout be on the Jupyter cell
#  and not the Terminal console).
# Available backends are 'loky', 'multiprocessing' and 'threading',
# startup default being 'loky'.
xgb.XGBRegressor(
        n_estimators=1
        , n_jobs=multiprocessing.cpu_count()-1
        , verbosity = 0
).fit(pd.DataFrame({"feature":[0]}), [0], verbose=0)

# @see Jupyter bug : 'https://stackoverflow.com/questions/55955330/'
# (printed-output-not-displayed-when-using-joblib-in-jupyter-notebook)
# REMARK : sklearn.GridSearchCV.fit uses joblib.Parallel when n_jobs>1


#/////////////////////////////////////////////////////////////////////////////////////


def grouped_mean_stdv(evaluations_dataframe_, val_rmse=True) :
    cv_folds_count = max(evaluations_dataframe_['model_cv_fold']+1)

    if val_rmse :
        column_name_prefix = 'val_'
    else :
        column_name_prefix = 'train_'

    def mean_stdv_cond(df_group):
        """
        if any 'cv_fold' stopped (due to early_stopping when learning stalls
        [i.e. when performance metrics stalls on the validation dataset]),
        return None as mean and as stdv
        When the standard behavior (unsatisafctory in the herein context) 
        of aggregated computation is :
            - keeps computing 'mean' across all training steps,
              even if only on cv_fold remains (when the other folds have been interrupted
              because they met conditions for early_stopping).
            - keeps computing 'stdv' across all training steps
              as long as models are still being trained on at least 2 cv folds.
        """

        if df_group['model_cv_fold'].count() >= cv_folds_count:
            return df_group[column_name_prefix+'rmse'].mean() \
                    , np.std(df_group[column_name_prefix+'rmse'],ddof=0)
        return None, None


    gdf = pd.DataFrame(
        evaluations_dataframe_.groupby(['n_estimators', 'model_params']) \
            .apply(mean_stdv_cond)
    ).reset_index(level=(0,1))
    #print(gdf.sort_values(by='model_params'))

    return pd.DataFrame(
            [ [row['n_estimators'], row['model_params'], row[0][0], row[0][1]]
             for index, row in gdf.iterrows()]
        ).dropna(axis=0).rename(columns={0:'n_estimators', 1:'model_params'
                                         , 2:column_name_prefix+'rmse_mean'
                                         , 3:column_name_prefix+'rmse_stdv'})


#/////////////////////////////////////////////////////////////////////////////////////


def xgbR_gridsearch_cv_agg(
    xgb_R_evaluations_df
) :
    """
    Aggregates result of GridSearchCV validation measures
    across the CV folds.

    Parameters :
        - xgb_R_evaluations_df (pandas.DataFrame) :
            as generated internally by 'my_NLP_RNN_fr_lib.my_xgboost.xgbR_gridsearch_cv',
            meaning :a pandas.DataFrame of the rmse measures
                - on each cv_fold
                - at each n_steps
                - of each set of hyperparameter values to be evaluated
            having 5 columns, namely :
                - n_estimators (int)
                - val_rmse (float)
                - train_rmse (float)
                - model_params (string)
                - model_cv_fold (int)

    Result :
        - fold_aggregated (pandas.DataFrame) :
            cv_folds_count * candidates_count * int(n_estimators/xgbR_gridsearch_cv.n_steps) rows
            [rough estimate, since measures are also taken at 'step 0' and 'step final'
             of each model instance training],
            6 columns, namely :
                - n_estimators (int) :
                    step at which the measure has been taken
                - model_params
                    the set of hyperparameter values corresponding to
                    the model being evaluated
                - with regards to the validation dataset :
                    - val_rmse_mean (float)
                    - val_rmse_stdv (float)
                - with regards to the training dataset :
                    - train_rmse_mean (float)
                    - train_rmse_stdv (float)
        - params_ranks (pandas.DataFrame) :
            sorted by descending rank, is comprised of 'candiates_count' rows
            and 3 columns, namely, for each xgbR model candidate :
                - model_params (str) :
                    a string representation of the dictionnary
                    comprising 'hyperparameter name-value' pairs
                    for each of the hyperparameters tuned.
                - val_min_rmse_mean (float) :
                    across the cv folds, the mean 'rmse' best value
                    of the validation datasets.
                - val_best_rmse_rank (int) :
                    the respective ranking of the set of hyperparameter
                    values (amond all candidates).
        - model_ids (pandas.DataFrame) :
            each xbg regressor model instance is assigned a unique ID
            (one per set of hyperparameter values per cv_fold),
            has 3 columns, namely :
                - model_params (str) :
                    a string representation of the dictionnary
                    comprising 'hyperparameter name-value' pairs
                    for each of the hyperparameters tuned.
                - model_cv_fold (int)
                - model_uid (int)
    """

    # 1)
    #################################################################################
    ## computing respective performance of the 'candidate' sets of hyperparameters ##
    ## mean and stdv across 'cv_folds'.                                            ##
    #################################################################################
    fold_aggregated_df = \
        grouped_mean_stdv(xgb_R_evaluations_df)
    fold_aggregated_df[['train_rmse_mean', 'train_rmse_stdv']] = \
        grouped_mean_stdv(xgb_R_evaluations_df, val_rmse=False
                         )[['train_rmse_mean', 'train_rmse_stdv']]
    #################################################################################
    # xgb_R_params_evaluations_df.columns :
    #     ['index', 'n_estimators', 'model_params', 'val_rmse_mean', 'val_rmse_stdv']
    #################################################################################
    #print("xgb_R_params_evaluations_df : " + chr(10) + str(xgb_R_params_evaluations_df) + chr(10))


    # 2)
    #################################################################################
    ## ranking sets of hyperparameters by best 'cv_folds models mean performance'. ##
    #################################################################################
    params_ranks_df = \
        fold_aggregated_df.groupby(['model_params'], as_index=False) \
            ['val_rmse_mean'].min().rename(columns={'val_rmse_mean': 'val_min_rmse_mean'})
    params_ranks_df['val_best_rmse_rank'] = \
        params_ranks_df['val_min_rmse_mean'].rank(ascending = True, method = 'min').astype(int)
    params_ranks_df = \
        params_ranks_df.sort_values(by=['val_min_rmse_mean'], ascending=True) \
        .reset_index(drop=True)
    #################################################################################
    # params_ranks_df.columns (sorted by descending rank) :
    #     ['model_params', 'val_min_rmse_mean', 'val_best_rmse_rank']
    #################################################################################
    #print("params_ranks_df : " + chr(10) + str(params_ranks_df) + chr(10)
    #      , file=sys.stdout, end = '\n', flush = True)


    return fold_aggregated_df, params_ranks_df


#/////////////////////////////////////////////////////////////////////////////////////


import matplotlib.pyplot as plt


#/////////////////////////////////////////////////////////////////////////////////////


def xgbR_gridsearch_cv_plot(
    xgbR_gridsearch_cv_model_instance
    , params_subplots = False
    , include_train_rmse = False
    , fig_width = 17
    , subplots_hspace = .5
    , restrict_rank = -1
) :
    """
    Plots result of 'xgbR_gridsearch_cv' validation measures.

    Parameters :
        - xgbR_gridsearch_cv_model_instance (my_NLP_RNN_fr_lib.my_xgboost.xgbR_gridsearch_cv_model) :
            as generated by 'my_NLP_RNN_fr_lib.my_xgboost.xgbR_gridsearch_cv'
        - params_subplots (boolean) :
            whether or not each set of hypeparameter
            values shall be printed on a separate subplot.
        - include_train_rmse (boolean) :
            whether or not include the measures reported
            from the the training dataset on the plots.
        - fig_width (int) :
            matplotlib figure width
        - restrict_rank (int) :
            strictly positive or ignored. Limit to be put on
            how many sets of hyperparameters shall be considered
            (from the best to the worst performer).

    Result :
        - fig (matplotlib.figure.Figure)
    """

    # 1)
    #######################################################################
    ## respective performance of the 'candidate' sets of hyperparameters ##
    ## mean and stdv across 'cv_folds'.                                  ##
    #######################################################################
    xgb_R_params_evaluations_df = \
        xgbR_gridsearch_cv_model_instance.fold_aggregated_df
    #######################################################################
    # xgb_R_params_evaluations_df.columns :
    #     ['n_estimators', 'model_params', 'val_rmse_mean', 'val_rmse_stdv']
    #######################################################################
    #print("xgb_R_params_evaluations_df : " + chr(10) + str(xgb_R_params_evaluations_df) + chr(10))


    # 2)
    #######################################################################
    ## ranking of the different sets of hyperparameters                  ##
    ## by best 'cv_folds models mean performance'.                       ##
    #######################################################################
    params_ranks = xgbR_gridsearch_cv_model_instance.params_ranks_df
    #######################################################################
    # params_ranks.columns (sorted by descending rank) :
    #     ['model_params', 'val_min_rmse_mean', 'val_best_rmse_rank']
    #######################################################################

    # 3)
    #######################################################################
    ## assigning each xbg regressor model instance a unique ID           ##
    ## (one per set of hyperparameter values per cv_fold).               ##
    #######################################################################
    model_ids = \
        xgbR_gridsearch_cv_model_instance.evaluations_df[['model_params', 'model_cv_fold']] \
        .drop_duplicates().reset_index(drop=True, level=0).reset_index(level=0) \
        .rename(columns={'index': 'model_uid'})
    #######################################################################
    # model_ids_df.columns :
    #     ['model_params', 'model_cv_fold', 'model_uid']
    #######################################################################


    if restrict_rank > 0 :
        restrict_rank = min(restrict_rank, params_ranks.shape[0])
        params_subplots = params_subplots and restrict_rank > 1
        params_ranks = params_ranks.iloc[range(restrict_rank)].reset_index(drop=True)
        # filter models, keep only ids with performance within 'restrict_rank' range
        # if "params_subplots", also sort those models,
        # so as to traverse the collection in the right order
        # for proper color and legend =>
        if not params_subplots :
            model_ids = \
                model_ids.merge(params_ranks, on=['model_params']) \
                .sort_values(by=['val_best_rmse_rank'], ascending=True) \
                .reset_index(drop=True) \
                .drop(['val_min_rmse_mean', 'val_best_rmse_rank'], axis=1)
        else :
            model_ids = \
                model_ids.merge(params_ranks, on=['model_params']) \
                .reset_index(drop=True) \
                .drop(['val_min_rmse_mean', 'val_best_rmse_rank'], axis=1)
    #print("params_ranks : " + chr(10) + str(params_ranks) + chr(10))
    #print("model_ids : " + chr(10) + str(model_ids) + chr(10))

    evaluations_df = \
        xgbR_gridsearch_cv_model_instance.evaluations_df \
        .merge(model_ids, on=['model_params', 'model_cv_fold'])
    #print("evaluations_df : " + str(evaluations_df.shape) + " (" + str(evaluations_df.columns) + ")")


    cmap = plt.cm.get_cmap('gist_rainbow', (params_ranks.shape[0]))
    nrows = (params_ranks.shape[0] if params_subplots else 1)
    fig, axes = plt.subplots(figsize=(fig_width, nrows*fig_width/8)
                             , ncols=2, nrows=nrows
                             #, sharex = True
                             #, sharey = True
                            )
    axes = axes if params_subplots else axes.reshape(1, axes.shape[0])
    #print(axes.shape)


    ############################################################################
    ## plot detailled chart (one curve per cv_fold per set of hyperparameter) ##
    ############################################################################
    for i in range(model_ids.shape[0]) :

        gridCv_model = evaluations_df[evaluations_df['model_uid']==model_ids['model_uid'][i]]
        #print(gridCv_model)
        #ax.plot(gridCv_model[['n_estimators']], gridCv_model[['train_rmse']])#, 'val_rmse']])
        model_param_str = str(gridCv_model['model_params'].values[0])
        #print(model_param_str + " - " + str(np.where(params_ranks['model_params'] == model_param_str)))
        model_param_idx = np.where(params_ranks['model_params'] == model_param_str)[0][0]

        details_ax = axes[model_param_idx][0] if params_subplots else axes[0][0]
        p = details_ax.plot(
            gridCv_model['n_estimators'], gridCv_model['val_rmse']
            , label = model_param_str
            , color = cmap(model_param_idx)
            , linestyle='solid' if not include_train_rmse
                                else (0, (5, 10)) # 'loosely dashed', @see 'https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html'
        )
        if include_train_rmse :
            p = list([p, details_ax.plot(gridCv_model['n_estimators'], gridCv_model['train_rmse']
                                         , label = model_param_str
                                         , color = cmap(model_param_idx)
                                        )
                     ])

    if not params_subplots :
        #legend, do not show duplicate entries (cv_folds)
        axes[0][0].legend() ; legend_lines = [] ; legend_labels = []
        cv_folds_count = max(model_ids['model_cv_fold'])+1
        for row in  [(x,y) for (i, (x,y)) in enumerate(zip(*axes[0][0].get_legend_handles_labels()))
                     if i%cv_folds_count == 0 and i < max(evaluations_df['model_uid']) ] :
            legend_lines.extend([row[0]])
            legend_labels.extend([row[1]])
        axes[0][0].legend(legend_lines, legend_labels, loc="upper right")


    ############################################################################
    ##       plot aggregated chart (one curve per set of hyperparameter)      ##
    ############################################################################
    for i in range(len(params_ranks)) :
        model_param_str = params_ranks['model_params'].iloc[i]
        #print(model_param_str)
        params_n_estimators = \
            xgb_R_params_evaluations_df[xgb_R_params_evaluations_df['model_params'] == model_param_str
                                       ]['n_estimators']
        params_val_rmse_mean = \
            xgb_R_params_evaluations_df[xgb_R_params_evaluations_df['model_params'] == model_param_str
                                       ]['val_rmse_mean']
        params_val_rmse_stdv = \
            xgb_R_params_evaluations_df[xgb_R_params_evaluations_df['model_params'] == model_param_str
                                       ]['val_rmse_stdv']
        aggregate_ax = axes[i][1] if params_subplots else axes[0][1]
        p = aggregate_ax.plot(
            params_n_estimators
            , params_val_rmse_mean
            , color=cmap(i)
            , linestyle='solid' if not include_train_rmse
                                else (0, (5, 10))
        )
        p = list([p, aggregate_ax.fill_between(
            params_n_estimators
            , params_val_rmse_mean + params_val_rmse_stdv
            , params_val_rmse_mean - params_val_rmse_stdv
            , facecolor=cmap(i)
            , alpha = 0.08 if not include_train_rmse
                           else .05
            , antialiased=True
            , edgecolor = cmap(i)
            , linewidth = 0 if not include_train_rmse else 6
            , linestyle = (0, (5, 10))
        )
                 ])
        if include_train_rmse :
            params_train_rmse_mean = \
                xgb_R_params_evaluations_df[xgb_R_params_evaluations_df['model_params'] == model_param_str
                                           ]['train_rmse_mean']
            params_train_rmse_stdv = \
                xgb_R_params_evaluations_df[xgb_R_params_evaluations_df['model_params'] == model_param_str
                                           ]['train_rmse_stdv']
            p = list([p, aggregate_ax.plot(
                params_n_estimators
                , params_train_rmse_mean
                , color=cmap(i)
                , linestyle='solid'
            )
                     ])
            p = list([p, aggregate_ax.fill_between(
                params_n_estimators
                , params_train_rmse_mean + params_train_rmse_stdv
                , params_train_rmse_mean - params_train_rmse_stdv
                , facecolor=cmap(i)
                , alpha = 0.08
            )
                 ])
        #aggregate_ax.set_title(model_param_str) # DEBUG, check against mis-alignments

    # harmonize xaxis and yaxis across subplots
    # (and yet have xthicklabels and ythicklabels displayed on them all,
    #  as opposed to when using 'plt.subplots(sharex, sharey)')
    axes[0][0].get_shared_x_axes().join(*axes.reshape(-1))
    axes[0][0].get_shared_y_axes().join(*axes.reshape(-1))
    axes[0][0].set_ylim(0, axes[0][0].get_ylim()[1])

    if params_subplots :
        for row_nb, ax in enumerate(axes[:,0]) :
            ax.annotate(params_ranks['model_params'][row_nb] +
                        " val_min_rmse_mean = " + str(round(params_ranks['val_min_rmse_mean'][row_nb], 4)) +
                        " (ranked #" + str(params_ranks['val_best_rmse_rank'][row_nb]) + ")"
                        , xy=(1.1, 1.05), xytext=(0, 0)
                        , xycoords='axes fraction', textcoords='offset points'
                        , size='large'
                        , ha='center', va='baseline')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace = subplots_hspace, wspace=None)
    # plt.subplot_tool() # in regular non-notebook uses of Matplotlib

    return fig


#/////////////////////////////////////////////////////////////////////////////////////







































































