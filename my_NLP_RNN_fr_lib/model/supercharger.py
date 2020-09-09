import time, sys
import pandas as pd
import numpy as np

from my_NLP_RNN_fr_lib.model.hyperparameters import get_new_hyperparameters
from my_NLP_RNN_fr_lib.display_helper import format_vertical_headers

from tqdm import tqdm
import pickle

from IPython.display import HTML, display


#/////////////////////////////////////////////////////////////////////////////////////


def best_nlp_models_hyperparameters(
    xgb_regressor
    , best_models_count
    , batch_models_count
    , NLP_predicted_val_rmse_threshold
    , show_batch_box_plots = False
) -> pd.core.frame.DataFrame :
    """
    Iterative routine allowing to identify 'most promising'
    sets of NLP hyperparameter values, when evaluated
    by an XGBoost regressor.
    Each iteration (a.k.a. "WAVE") is made up of 'batch_models_count'
    sets of NLP hyperparameter values.
    The routine is interrupted :
        - either if the 'NLP_predicted_val_rmse_threshold' condition is met
        - or when the user commands it via a 'KeyboardInterrupt' event
          (smooth exit)

    Parameters :
         - xgb_regressor (xgboost.sklearn.XGBRegressor) :
             XGBoost regressor trained against an NLP models performance dataset.
         - best_models_count (int) :
             The number of 'predicted best performing'
             sets of NLP hyperparameters to be maintained and returned.
         - batch_models_count (int) :
             The number of randomly-generated set of NLP hyperparameters
             to be submitted to the XGBoost regressor for performance prediction
             at each iteration of the herein routine.
         - NLP_predicted_val_rmse_threshold (float) :
             value of predicted NLP performance measure, i.e. 'validation rmse'
             under which the herein routine shall automatically stop looping.
             When all "best_models_count" sets of NLP hyperparameters
             are predicted to perform better than that user-specified value,
             the routine is considered completed.
         - show_batch_box_plots (bool) :
             whether or not to display a 'predicted val_rmse' distribution boxplot
             for each "batch" (each loop).

    Result :
        - xgbR_best_hyperparameters (pandas.core.frame.DataFrame) :
            The 'predicted' best-performing sets of hyperparameters
            to be used when building an NLP model following
            the 'my_NLP_RNN_fr_lib.model.architecture.build_model' architecture.
    """

    def row_index_n_bold_style(row_index_min: int) :
        """
        return a style apply lambda function.
        If row index > row_index_min, the format of the cells
        is made green background & bold-weight font.
        """

        def new_best_row_bold_style(row):
            if row.name > row_index_min :
                styles = {col: "font-weight: bold; background-color: lightgreen;" for col in row.index}
            else :
                styles = {col: "" for col in row.index}

            return styles

        return new_best_row_bold_style


    xgbR_best_hyperparameters = pd.DataFrame(columns = ['spatial_dropout_prop', 'recurr_units',
           'recurrent_regularizer_l1_factor', 'recurrent_regularizer_l2_factor',
           'recurrent_dropout_prop', 'conv_units', 'kernel_size_1',
           'kernel_size_2', 'dense_units_1', 'dense_units_2', 'dropout_prop', 'lr',
           'val_rmse (predicted)'])

    
    def get_best_hyperparameters(
        xgbR_best_hyperparameters
    ) :
        """
        Parameters :
            - xgbR_best_hyperparameters (pd.DataFrame) :
                datafrm into which any new "best" set of NLP hyperparameters
                shall be inserted
        """
        ####################################
        ## get 'batch_models_count' fresh ##
        ## sets of hyperparameters values ##
        ####################################
        try:
            hyperparameters = get_new_hyperparameters(verbose = 2, models_count = batch_models_count)
        except TimeoutError as e :
            print(e)
            raise e
        xgbR_X_new = hyperparameters.drop([
            'lr_decay_rate', 'lr_decay_step' ], axis=1)
        print("", end='\n', file=sys.stdout, flush=True)


        #######################################
        ## get the associated                ##
        ## predicted NLP models performances ##
        #######################################
        print(f"Predict models performance :", end='\n', file=sys.stderr, flush=True)
        tic = time.perf_counter()

        chunk_size = 50_000
        xgbR_y_pred = []
        for i in tqdm(range(-(-xgbR_X_new.shape[0]//chunk_size))
                      , bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:45}{r_bar}') :
            indices_slice = list(range(i*chunk_size, min((i+1)*chunk_size, xgbR_X_new.shape[0])))
            #print("["+str(indices_slice[0])+"-"+str(indices_slice[-1])+"] - " + str(len(indices_slice)))
            xgbR_y_pred = \
                np.append(xgbR_y_pred
                          , xgb_regressor.predict(xgbR_X_new.iloc[indices_slice]))

        toc = time.perf_counter()
        print(f"Predicted performance of {xgbR_X_new.shape[0]:,} NLP models  in {toc - tic:0.4f} seconds"
              , end='\n', file=sys.stderr, flush=True)


        #######################################
        ## plot the distribution             ##
        ## among the 'batch_models_count'    ##
        ## predicted NLP models performances ##
        #######################################
        if show_batch_box_plots :
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111)
            ax.boxplot(xgbR_y_pred
                       , showfliers=False # not plotting outliers
                      )
            #ax.set_ylim(max(ax.get_ylim()[0], .5), ax.get_ylim()[1])
            ax.set_title("NLP Model performance (rmse)")
            plt.show()


        ##########################################################
        ## isolate the sets of hyperparameters values           ##
        ## associated to best predicted NLP models performances ##
        ##########################################################
        xgbR_hyperparameters = xgbR_X_new
        xgbR_hyperparameters["val_rmse (predicted)"] = xgbR_y_pred
        xgbR_hyperparameters = \
            xgbR_hyperparameters.sort_values('val_rmse (predicted)', ascending=True)

        print("Processed " + "{:,}".format(xgbR_hyperparameters.shape[0]) + " new random sets of hyperparameter values " +
              "with below distribution of predicted NLP models rmse :"
              , end='\n', file=sys.stdout, flush=True)
        #format_vertical_headers(xgbR_hyperparameters[:best_models_count]) # "best_models_count" best "new"
        display(
            HTML(pd.DataFrame(xgbR_y_pred).describe(percentiles = [0.0001, 0.01,.25,.5,.75]).T \
                 .to_html(index=False))
        )


        # offset index, to make sure any value below "best_models_count"
        # is an historical "best" during the (following) merge operation =>
        xgbR_hyperparameters.index = \
            xgbR_hyperparameters.index + best_models_count

        # merge two dataframes (historical "best" and "batch_models_count" new)
        xgbR_best_hyperparameters = \
            pd.concat(
                [xgbR_hyperparameters
                , xgbR_best_hyperparameters]
                , axis=0
                , sort=False
            ).sort_values('val_rmse (predicted)', ascending=True)[:best_models_count]

        # indentify "new best" among "bests" by the "index" value of the rows
        new_best_count = xgbR_best_hyperparameters[
                xgbR_best_hyperparameters.index > (best_models_count - 1)
            ].shape[0]
        print(str(new_best_count) + " new \"best\" hyperparameter values"
              + (" :" if new_best_count > 0 else "")
              , end='\n', file=sys.stdout, flush=True)
        if new_best_count > 0 :
            format_vertical_headers(
                xgbR_best_hyperparameters.style.apply(
                    lambda df_row: row_index_n_bold_style(best_models_count-1)(df_row)
                    , axis=1)
            )
        xgbR_best_hyperparameters.reset_index(drop=True, inplace=True)

        return xgbR_best_hyperparameters

    loops_completed_count = 0
    try:
        ######################################
        ## loop until we have collected     ##
        ## "best_models_count" sets         ##
        ## of "best" hyperparameters values ##
        ######################################
        while (
            (xgbR_best_hyperparameters.shape[0] < best_models_count) |
            (xgbR_best_hyperparameters['val_rmse (predicted)'] > NLP_predicted_val_rmse_threshold).any()
        ) :

            print("WAVE #" + str(loops_completed_count+1), end='\n', file=sys.stdout, flush=True)
            xgbR_best_hyperparameters = \
                get_best_hyperparameters(xgbR_best_hyperparameters)

            print("", end='\n', file=sys.stdout, flush=True)
            loops_completed_count += 1
            #break

    except KeyboardInterrupt:
        sys.stderr.flush()
        print('\x1b[4;33;41m' + ' Interrupted by user ! ' + '\x1b[0m', end='\n', file=sys.stdout, flush=True)

    print('\033[1m' + '\033[4m' + '\033[96m' +
          "As per the prediction of the XGBoost regressor, " +
          f"out of the {loops_completed_count*batch_models_count:,} random sets of hyperparameters considered, " +
          f"here are the final {xgbR_best_hyperparameters[:best_models_count].shape[0]:,} predicted \"best\" :" +
          '\033[0m'
          , end='\n', file=sys.stdout, flush=True)
    format_vertical_headers(xgbR_best_hyperparameters[:best_models_count])

    return xgbR_best_hyperparameters


#/////////////////////////////////////////////////////////////////////////////////////






































































