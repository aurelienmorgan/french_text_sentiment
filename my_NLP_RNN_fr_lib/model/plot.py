import os, sys
import numpy as np
import pandas as pd

import cv2

import dill as pickle

import matplotlib.pyplot as plt


#/////////////////////////////////////////////////////////////////////////////////////


from keras.utils import model_to_dot

def plot_model(
    keras_model
    , bgcolor, forecolor
    , framecolor, watermark_text
    , fillcolors, fontcolors
) -> np.ndarray :
    """
    Parameters :
        - keras_model (keras.engine.training.Model) :
            the model the architecture of which is to be plotted.
        - bgcolor (str) :
            the hexadecimal expression of the background color.
        - forecolor (str) :
            the hexadecimal expression of the foreground color.
        - framecolor (str) :
            the hexadecimal expression of the color
            of the image border.
        - watermark_text (str) :
            the text to be watermarked at the bottom-right corner
            of the image.
        - fillcolors (dic) :
            a dictionnary with entries ('node name': 'fill hex color')
            for nodes to be filled with a color other than transparent.
        - fontcolors (dic) :
            a dictionnary with entries ('node name': 'font hex color')
            for nodes to be labell in a color other than 'forecolor'.

    Resuts :
        - an 'np.ndarray' of 3 colors channels
          and dimension (height x width)
    """

    graph = model_to_dot(keras_model, show_layer_names=False)
    graph.set_bgcolor(bgcolor)

    nodes = graph.get_node_list() ; edges = graph.get_edge_list()

    for node in nodes:
        if node.get_name() == 'node' :
            node.obj_dict['attributes'] = \
                {'shape': 'record', 'style': "filled, rounded"
                 , 'fillcolor': bgcolor, 'color': forecolor
                 , 'fontcolor': forecolor, 'fontname': 'helvetica'}
        #print(str(node.get_label()) + " - " + str((node is not None) & (node.get_label() in colors)))
        if node.get_label() in fillcolors :
            node.set_fillcolor(fillcolors[node.get_label()])
        if node.get_label() in fontcolors :
            node.set_fontcolor(fontcolors[node.get_label()])
    # 'graph.set_edge_defaults' will fail (defaults being appended last) =>
    for edge in edges: # apply successively to each edge
        edge.set_color(forecolor) ; edge.set_arrowhead("vee")

    #print(graph.to_string())
    tmp_filename = os.path.join('tmp', 'colored_tree.png')
    graph.write_png(tmp_filename)
    #from IPython.core.display import display, Image ; display(Image(graph.create_png()))

    image = cv2.cvtColor(cv2.imread(tmp_filename), cv2.COLOR_BGR2RGB)
    os.remove(tmp_filename)

    def hex_to_rgb(hex_color) : return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    #watermark
    texted_image = cv2.putText(img=np.copy(image), text=watermark_text
                , org=(image.shape[1]-(10+int(7.33*len(watermark_text))), image.shape[0]-10)
                , fontFace=cv2.FONT_HERSHEY_COMPLEX
                , fontScale=.4, color=hex_to_rgb(framecolor), lineType=cv2.LINE_AA, thickness=1)
    # padding
    outer_pad_top, outer_pad_bot, outer_pad_left, outer_pad_right = 4, 4, 4, 4 ; outer_pad_color = hex_to_rgb(bgcolor)
    inner_pad_top, inner_pad_bot, inner_pad_left, inner_pad_right = 2, 2, 2, 2 ; inner_pad_color = hex_to_rgb(framecolor)
    padded_image = cv2.copyMakeBorder(
        cv2.copyMakeBorder(texted_image
                           , inner_pad_top, inner_pad_bot, inner_pad_left, inner_pad_right
                           , borderType=cv2.BORDER_CONSTANT, value=inner_pad_color)
        , outer_pad_top, outer_pad_bot, outer_pad_left, outer_pad_right
        , borderType=cv2.BORDER_CONSTANT, value=outer_pad_color)

    return cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR)


#/////////////////////////////////////////////////////////////////////////////////////


def plot_trained_models_history(
    trained_models_df
    , validation_only = False
    , labels = None
) -> None :
    """
    Displays plots of (stacked) models training history.

    Parameters :
        - trained_models_df (DataFrame) :
            DataFrame with nrwos of trained models.
            The ["local_path"] column is used to locate the training history
            locally stored pickle files.
        - validation_only (bool) :
            whether or not to only plot curves drawn from the 'validation' dataset
            (as opposed to the default behavior consisting in
            also plotting curves from the 'development' dataset)
        - labels (list) :
            list of length 'trained_models_df.shape[0]' of the label
            to be used for each curve, respectively.

    Results :
        - N/A
    """

    fig = plt.figure(figsize=(14,8))
    host1 = fig.add_subplot(2, 1, 1) ; host1.set_title('Training Loss')
    par1 = host1.twinx()
    host2 = fig.add_subplot(2, 1, 2) ; host2.set_title('Training Performance')
    par2 = host2.twinx()

    history_df = None

    colors_count = min(10, trained_models_df.shape[0]+1)
    colors = plt.cm.get_cmap('gist_rainbow', (colors_count)) # cmap

    for i, trained_model in trained_models_df.iterrows() :
        pickle_path = \
            os.path.join(trained_model["local_path"]
                         , "train_hstory_" + trained_model["timestamp"] + ".pickle")
        if not os.path.isfile(pickle_path) :
            sys.stderr.write('Model history for model %s missing \n'
                             % trained_model["timestamp"])
        else :
            with open(pickle_path, 'rb') as f:
                history_reloaded = pd.DataFrame(pickle.load(f))

            color = colors(i % colors_count)
            idx_str = str(i) if labels is None else labels[i]
            if not validation_only :
                host1.plot(history_reloaded.index, history_reloaded['loss']
                           , label='training ' + idx_str
                           , color=color, linewidth=.7)
            host1.plot(history_reloaded.index, history_reloaded['val_loss']
                       , label='validation ' + idx_str
                       , color=color, linewidth=2)
            par1.plot(history_reloaded.index, history_reloaded['lr']
                      , label='learning rate ' + idx_str
                      , color=color, linestyle='dotted', linewidth=.6)

            if not validation_only :
                host2.plot(history_reloaded.index, history_reloaded['root_mean_squared_error']
                           , label='training ' + idx_str
                           , color=color, linewidth=.7)
            host2.plot(history_reloaded.index, history_reloaded['val_root_mean_squared_error']
                       , label='validation ' + idx_str
                       , color=color, linewidth=2)
            par2.plot(history_reloaded.index, history_reloaded['lr']
                      , label='learning rate ' + idx_str
                      , color=color, linestyle='dotted', linewidth=.6)

    host1.set_xlabel('Epoch'); host1.set_ylabel('Loss')
    host1.set_ylim([ 0., min(150, host1.get_ylim()[1]) ])
    host1_legend = host1.legend(loc='upper right')
    par1.set_ylabel('Learning Rate')
    par1.legend(loc='upper right', bbox_to_anchor=(0., 0, 0.7, 1.))

    host2.set_xlabel('Epoch'); host2.set_ylabel('RMSE')
    host2.set_ylim([ (.5 if validation_only else .4), min(1., host2.get_ylim()[1]) ])
    host2_legend = host2.legend(loc="upper right")
    par2.set_ylabel('Learning Rate')
    par2.legend(loc='upper right', bbox_to_anchor=(0., 0, 0.7, 1.))

    fig.tight_layout()
    plt.show()


#/////////////////////////////////////////////////////////////////////////////////////


import numpy as np
from math import ceil
import matplotlib.colors as colors
from matplotlib.transforms import Bbox
import copy


def hyperparameters_univariate_rmse_plot(
    trained_models_df
    , col_name = 'best_val_rmse'
    , outliers_threshold = None
    , subplots_ncols = 6
    , figsize = (16, 8)
) -> None :
    """
    Parameters :
        - trained_models_df (pandas.DataFrame) :
            set of hyperparameter values and associated performance metrics.
            One NLP model per row.
        - col_name (str) :
            column to be considered for 'best NLP model' ranking criteria.
        - outliers_threshold (float) :
            datapoints for which 'col_name' exceeds this value are to be
            outside the plots window (thus not displayed)
        - subplots_ncols (int) :
            number of columns of the figure subplots grid.
        - figsize (tuple(int)) :
            size of the matplotlib figure to be returned.
    """

    models_count = trained_models_df.shape[0]

    if outliers_threshold :
        trained_models_df = copy.deepcopy(
            trained_models_df[trained_models_df[ col_name ] < outliers_threshold] \
                .reset_index(drop=True)
        )

    best_model_pos =  np.argmin(np.array(trained_models_df[ col_name ]))

    hyperparameters_list = ['spatial_dropout_prop', 'recurr_units',
       'recurrent_regularizer_l1_factor', 'recurrent_regularizer_l2_factor',
       'recurrent_dropout_prop', 'conv_units', 'kernel_size_1', 'kernel_size_2',
       'dense_units_1', 'dense_units_2', 'dropout_prop', 'lr']

    subplots_nrows = ceil(len(hyperparameters_list) / subplots_ncols)
    fig, axes = plt.subplots(figsize=figsize, nrows=subplots_nrows
                             , ncols=subplots_ncols)

    fig.suptitle('Distribution of Performance among ' + str(models_count) +
                 ' Trained NLP Models', size=20)

    color_map = colors.LinearSegmentedColormap.from_list(
        'unevently divided'
        , [(0, 'blue'), (.01, 'green'), (.1, 'yellow'), (1, 'red')]
        #, [(0, 'blue'), (.01, 'green'), (.5, 'yellow'), (1, 'red')]
    )
    
    lightgray_color_map = colors.LinearSegmentedColormap.from_list(
        "lightgray", ["lightgray", "lightgray"])

    for i, hyperparameter in enumerate(hyperparameters_list) :
        im = axes.flatten()[i].scatter(trained_models_df[hyperparameter]
                                        , trained_models_df[ col_name ]
                                       , c = trained_models_df[col_name ]
                                       , cmap = color_map
                                       , edgecolors = 'lightgray', linewidth = .2
                                      )
        axes.flatten()[i].set_xlabel(
            ' '.join(['$\\bf{'+i.replace('_', '\_')+'}$' for i in hyperparameter.split(' ')])
        ) # bold xlabel
        if i % subplots_ncols == 0: axes.flatten()[i].set_ylabel('best validation rmse')
            # highlight THE best-performing model in blue
        axes.flatten()[i].plot(trained_models_df[hyperparameter].iloc[best_model_pos]
                                , trained_models_df[ col_name ].iloc[best_model_pos]
                                , 'bo')


    # delete empty axes (if "nrows*ncols > parent_hyperparameter bins_count")
    for i in range(subplots_nrows*subplots_ncols)[len(hyperparameters_list):] :
        fig.delaxes(axes.flatten()[i])

    fig.tight_layout()
    fig.subplots_adjust(top=0.87) # allow room for global title

    plt.draw()

    # add horizontal/top colorbar
    p_left = axes.flatten()[0].get_position().get_points().flatten()
    p_right = axes.flatten()[subplots_ncols-1].get_position().get_points().flatten()
    ax_cbar0 = fig.add_axes([p_left[0], .93
                             , p_right[2]-p_left[0], 0.01]) # rect [left, bottom, width, height]
    plt.colorbar(im, cax=ax_cbar0, label = 'best validation rmse', orientation='horizontal')


    plt.show()

    return fig


#/////////////////////////////////////////////////////////////////////////////////////


def hyperparameter_rmse_boxplot(
    trained_models_df
    , hyperparameter
    , ax
    , bins_count = 0
    , bins_edges = None
    , bin_edge_format="{0:.2f}"
    , xlabels_ha="center"
    , col_name = 'best_val_rmse'
) :
    """
    Draws boxplots of 'col_name' with regards to the values taken by 'hyperparameter'.

    Arguments :
        - trained_models_df (pandas.DataFrame) :
            set of hyperparameter values for 'nrows' trained models
        - hyperparameter (string) :
            name of the hyperparameter against which
            to draw the ditribution of values taken by 'col_name'
        - ax (matplotlib.Axes) :
            object to hold the drawn chart
        - bins_count (int) :
            number of bins to use to draw the boxplots.
            If bins_count=1, one single boxplot depicting the distribution of
            values taken by 'col_name' is drawwn, notwithstanding the ditribution of
            values taken by 'hyperparameter'. The behavior of the function is then
            equivalent to that of 'matplotlib.pyplot.boxplot'.
        - xlabels_ha (string) :
            alignment of the x-labels with regards to their respective ticks
            (can be either 'left', 'center' or 'right')
        - bin_edges (numpy.ndarray) :
            if not null, 'bins_count' is ignored. Bounds of the bins to be used.
        - bin_edge_format (string) :
            string formatting to be applied to bin edges (i.e. bin bounds)
            on the xaxis labels of the drawn chart.

    Results :
        - bin_edges (list) :
            the edges (i.e. bounds) of the bins of values taken by 'hyperparameter'
            which are used to draw the boxplots
        - grouped_medians (list) :
            the median values taken by 'col_name' for each of
            the bins of values taken by 'hyperparameter', respectively.
    """

    trained_models_df = \
        trained_models_df[[ hyperparameter, col_name ]].copy().reset_index(drop=True)

    if bins_edges is None :
        bins_edges = np.linspace(min(trained_models_df[hyperparameter])
                                 , max(trained_models_df[hyperparameter]), bins_count+1)
    else :
        bins_count = len(bins_edges) - 1
    #print( bins_edges )

    def bin_number(x, bins_edges = bins_edges) :
        for i in range(1, len(bins_edges)) :
            if (x<=bins_edges[i]) : return i

    bin_number_arr = \
        np.array(trained_models_df[hyperparameter].apply(bin_number), dtype='int')
    #print(set(bin_number_arr))

    grouped_medians = trained_models_df.groupby(bin_number_arr).median()
    #print(( grouped_medians[ col_name ] ))

    ########################################################################################
    # add "empty" bins (bins of 'hyperparameter' values for which no 'col_name' exists
    # among the 'trained_models_df' dataframe)
    for bin_number in range(1, bins_count+1) :
        if not (bin_number in bin_number_arr) :
            #print("bin_number : " + str(bin_number))
            bin_center = (bins_edges[bin_number-1]+bins_edges[bin_number])/2
            #print("center : " + str(bin_center))
            bin_number_arr = np.append(np.array(bin_number_arr), [bin_number])

            trained_models_df.loc[trained_models_df.shape[0]] = [bin_center] + [float('NaN')]
    ########################################################################################

    xticklabels = \
        [ bin_edge_format.format(bins_edges[i])+"-"+bin_edge_format.format(bins_edges[i+1])
          for i in range(len(bins_edges)-1) ]
    #print(xticklabels)

    bxplt = trained_models_df.boxplot(
        column=[ col_name ]
        , by=bin_number_arr
        , ax=ax
        , grid=False
        , return_type='dict'
        , showfliers=False # not plotting outliers
        , showmeans=False
    )
    # boxplot style adjustments
    linewidth = 2
    [[item.set_linewidth(linewidth) for item in bxplt[key]['boxes']] for key in bxplt.keys()]
    [[item.set_color('#FFA500') for item in bxplt[key]['boxes']] for key in bxplt.keys()]
    [[item.set_color('#A9A9A9')  for item in bxplt[key]['medians']] for key in bxplt.keys()]
    [[item.set_linewidth(linewidth) for item in bxplt[key]['whiskers']] for key in bxplt.keys()]
    [[item.set_color('#03414e') for item in bxplt[key]['whiskers']] for key in bxplt.keys()]
    [[item.set_linestyle('dotted') for item in bxplt[key]['whiskers']] for key in bxplt.keys()]
    [[item.set_linewidth(linewidth) for item in bxplt[key]['caps']] for key in bxplt.keys()]
    [[item.set_color('#03414e') for item in bxplt[key]['caps']] for key in bxplt.keys()]
    [[item.set_linestyle('dotted') for item in bxplt[key]['caps']] for key in bxplt.keys()]
    #border
    ax.spines['left'].set_color('#d3d3d3')
    ax.spines['bottom'].set_color('#d3d3d3')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.patch.set_facecolor('#DF00FE')
    ax.patch.set_alpha(0.02)

    forecolor = '#777777'
    ax.set_xlabel(hyperparameter)#, color=forecolor)
    ax.set_ylabel( col_name )#, color=forecolor )
    ax.set_title(None) ; ax.get_figure().suptitle('')
    ax.tick_params(axis='y', colors=forecolor)
    ax.tick_params(axis='x', colors=forecolor)
    ax.set_xticklabels(xticklabels, color=forecolor, rotation = 45, ha=xlabels_ha)

    ax.plot( grouped_medians[ col_name ], c='#4E7566' )
    grouped_medians.columns = \
        [ hyperparameter + " (bin center)", col_name + ' (bin median)' ]


    return (bins_edges, grouped_medians)


#/////////////////////////////////////////////////////////////////////////////////////


from matplotlib.figure import Figure


def stratified_hyperparameter_rmse_boxplot(
    trained_models_df
    , parent_hyperparameter, parent_strats_count, parent_strat_edge_format
    , hyperparameter, bins_count, bin_edge_format
    , col_name = 'best_val_rmse'
    , ncols = 3
    , figsize=(12, 30)
) -> Figure :
    """
    Draws several subplots, each of which consists in a chart of boxplots of 'col_name'
    with regards to the values taken by 'hyperparameter'.
    The subplotting corresponds to different strats of values of 'parent_hyperparameter'.

    Arguments :
        - trained_models_df (pandas.DataFrame) :
            set of hyperparameter values for 'nrows' trained models

        - parent_hyperparameter (string) :
            name of the parent_hyperparameter to stratify.
        - parent_strats_count (int) :
            number of strats to consider for the range of values
            taken by parent_hyperparameter.
        - parent_strat_edge_format (string) :
            string formatting to be applied to strat edges (i.e. strat bounds)
            on the subplot titles of the drawn chart.

        - hyperparameter (string) :
            name of the hyperparameter against which
            to draw the ditribution of values taken by 'col_name'
        - bins_count (int) :
            number of bins to use to draw the boxplots.
            If bins_count=1, one single boxplot depicting the distribution of
            values taken by 'col_name' is drawwn, notwithstanding the ditribution of
            values taken by 'hyperparameter'. The behavior of the function is then
            a stratified equivalent to that of 'matplotlib.pyplot.boxplot'.
        - bin_edge_format (string) :
            string formatting to be applied to bin edges (i.e. bin bounds)
            on the xaxis labels of the drawn chart.

        - ncols (int) :
            number of column along which the subplots shall be arranged
            within the matplotlib figure.
        - figsize (tuple(int)) :
            width & height of the figure to be displayed.

    Results :
        - fig (matplotlib.figure.Figure)
            a drawn matplotlib figure object
    """

    df = trained_models_df[[parent_hyperparameter, hyperparameter, col_name]] \
            .copy().reset_index(drop=True)
    parent_strats_edges = np.linspace(min(df[parent_hyperparameter])
                                    , max(df[parent_hyperparameter]), parent_strats_count+1)
    def strat_number(x, strats_edges = parent_strats_edges) :
        for i in range(1, len(strats_edges)) :
            if (x<=strats_edges[i]) : return i
    df['groupby_col'] = np.array(df[parent_hyperparameter].apply(strat_number), dtype='int')
    #print(df['groupby_col'])

    # groupby 'parent_hyperparameter'
    parent_grouped = dict(list(
        df[['groupby_col', hyperparameter, col_name]].groupby(
            'groupby_col', as_index=False, sort=True)
    ))
    #print("max(grouped.keys()) : " + str(max(parent_grouped.keys())))

    nrows = -(-max(parent_grouped.keys()) // ncols) # rounded-up integer
    # set up a figure with 2 rows, 5 colums
    fig, ax = plt.subplots(nrows, ncols,
                           sharey=True,
                           figsize=figsize
                          )
    ax = ax.reshape(-1) # flatten, easy access to n-th element


    bins_edges = np.linspace(min(df[hyperparameter])
                             , max(df[hyperparameter]), bins_count+1)
    # iterate through our grouped and plot 
    for (k, v) in parent_grouped.items():
        hyperparameter_rmse_boxplot(v[[hyperparameter, col_name]], hyperparameter, ax[k-1]
                                    , bins_edges = bins_edges, bin_edge_format=bin_edge_format
                                    , col_name = col_name
                                   )
    # handle subplots titles here, to include cases of
    # "bins of parent_hyperparameter that are empty
    # (i.e. no such trained NLP model)"
    for k in range(max(parent_grouped.keys())+1) :
        #ax[k-1].set(title=f'{k}')
        ax[k-1].set_title(
            label = parent_hyperparameter + " in " +
            "[" + parent_strat_edge_format.format(parent_strats_edges[k-1]) +
            "-" + parent_strat_edge_format.format(parent_strats_edges[k]) + "]"
            , fontdict = {'fontsize':12, 'color':'blue'}
        )
        #ax[k-1].set_ylim([0.6,1.2])
        if ((k-1) % ncols != 0) : ax[k-1].set_ylabel(None)


    # delete empty axes (if "nrows*ncols > parent_hyperparameter bins_count")
    for i in range(nrows*ncols)[max(parent_grouped.keys()):] :
        fig.delaxes(ax[i])

    return fig


#/////////////////////////////////////////////////////////////////////////////////////


import re, io
from matplotlib.transforms import blended_transform_factory


def hyperparameters_univariate_rmse_overlay_plot(
    hyperparameters_univariate_rmse_fig
    , trained_models_overlay_df
    , col_name = 'best_val_rmse'
) -> Figure :
    """
    Overlays a dataframe on top of a figure plotted
    using the "hyperparameters_univariate_rmse_plot" method.

    Parameters :
        - hyperparameters_univariate_rmse_fig (Figure) :
            base figure to use as a grayed-out "background".
        - trained_models_overlay_df (pandas.DataFrame) :
            set of hyperparameter values and associated performance metrics.
            One NLP model per row.
        - col_name (str) :
            column to be considered for 'best NLP model' ranking criteria.
    """

    color_map = colors.LinearSegmentedColormap.from_list(
            'unevently divided'
            , [(0, 'blue'), (.3, 'green'), (.5, 'yellow'), (1, 'red')]
        )

    # REMARK : axes cannot be copied (even deepcopy will fail).
    # Hence to obtain a true copy of an axes object, we use 'pickle'.
    buf = io.BytesIO()
    pickle.dump(hyperparameters_univariate_rmse_fig, buf)
    buf.seek(0)
    xgboost_identified_nlp_models_fig = pickle.load(buf)


    latex_pattern = re.compile('\$.*{(.*)}\$')

    # retrieve "previous" min (from lowest y-value from scatter plot #0)
    previous_min = \
        min(
            [data_point[1]
             for data_point
             in xgboost_identified_nlp_models_fig.axes[0].get_children()[0].get_offsets()]
        )
    #print("previous_min : " + str(previous_min))

    # loop over all axes, except the colorbar
    for i in range(len(xgboost_identified_nlp_models_fig.axes)-1) :

        # matplotlib.collections.PathCollection object
        xgboost_identified_nlp_models_fig.axes[i].get_children()[0].set_facecolor('lightgray')

        hyperparameter = \
            latex_pattern.match(xgboost_identified_nlp_models_fig.axes[i].get_xlabel()
                               ).group(1).replace('\_', '_')
        #print(hyperparameter)

        # use 'blended' 'transformation' so as to
        # place the label at percentage point of the x-axis
        # (independent of scale, whatever the range of values of x covered)
        xaxis_transform = blended_transform_factory(xgboost_identified_nlp_models_fig.axes[i].transAxes
                                                    , xgboost_identified_nlp_models_fig.axes[i].transData)
        xgboost_identified_nlp_models_fig.axes[i].text(-.15, previous_min, 'random best'
                                                       , transform = xaxis_transform #, color = 'gray'
                                                      )

        xgboost_identified_nlp_models_fig.axes[i].axhline(
            y=previous_min, color='lightgray', linestyle='-', linewidth=.5, zorder=1)

        im = xgboost_identified_nlp_models_fig.axes[i].scatter(
            trained_models_overlay_df[hyperparameter]
            , trained_models_overlay_df[ col_name ]
            , c = trained_models_overlay_df[col_name ]
            , cmap = color_map
            , edgecolors = 'lightgray', linewidth = .2
            , zorder=2
        )

    # clear "previous" colorbar axis
    xgboost_identified_nlp_models_fig.axes[-1].cla()
    # add horizontal/top colorbar
    xgboost_identified_nlp_models_fig.colorbar(
        im, cax=xgboost_identified_nlp_models_fig.axes[-1]
        , label = 'best validation rmse (xgboost generated hyperparameters sets)'
        , orientation='horizontal')


    xgboost_identified_nlp_models_fig.suptitle(
        'Distribution of Performance among ' + str(trained_models_overlay_df.shape[0]) +
        ' XGBoost-identified NLP Models', size=20)


    # 'pickled' figures lack a canvas => create one
    def show_figure(fig: Figure) -> None:
        """
        create a dummy figure and use its
        manager to display "fig"
        (to be used when "fig" lacks one)
        """
        if not fig.canvas :
            dummy = plt.figure()
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = fig
            fig.set_canvas(new_manager.canvas)

    show_figure(xgboost_identified_nlp_models_fig)


    return xgboost_identified_nlp_models_fig


#/////////////////////////////////////////////////////////////////////////////////////


def warming_up_fine_tuning_history(
    warming_up_history, warming_up_lr
    , fine_tuning_history
    , fig1_size
    , fig2_size

    , epsilon=1e-07 # keras.backend.epsilon()
) -> Figure :
    """
    """

    try :
        warming_up_epochs = len(warming_up_history['loss'])
    except KeyError as kerr :
        raise KeyError('"warming_up_history.loss" column missing')

    ######################################################################
    ##       concatenate "warming-up" and "fine-tuning" histories       ##
    ######################################################################
    history_df = pd.concat(
        (
            pd.concat(
                (
                    pd.DataFrame(copy.deepcopy(warming_up_history))
                    , pd.DataFrame({'lr': [warming_up_lr]*warming_up_epochs})
                )
                , axis = 1)
            , pd.DataFrame(copy.deepcopy(fine_tuning_history))
        )
        , axis = 0).reset_index(drop=True)
    history_df.index -= warming_up_epochs

    column_names = [
            'val_loss', 'val_categorical_crossentropy', 'val_categorical_accuracy',
           'val_precision', 'val_recall', 'val_precision_NEG', 'val_recall_NEG',
           'val_precision_NEUT', 'val_recall_NEUT', 'val_precision_POS',
           'val_recall_POS', 'loss', 'categorical_crossentropy',
           'categorical_accuracy', 'precision', 'recall', 'precision_NEG',
           'recall_NEG', 'precision_NEUT', 'recall_NEUT', 'precision_POS',
           'recall_POS', 'lr']
    assert (~(~pd.DataFrame(column_names).isin(history_df.columns.values)).any())[0] \
        , "check the format of the input data. at least 1 column name is missing"
    assert (len(history_df.loc[:,(column_names)].select_dtypes(include=[np.number]).columns) ==
            len(history_df.loc[:,(column_names)].columns)) \
        , "check the format of the input data. at least 1 column contains non-numeric"
    ######################################################################


    ######################################################################
    ## calculate the "F1 SCORE" columns (from "PRECISION" and "RECALL") ##
    ## f1_score = 2*(precision*recall)/(precision+recall+epsilon)       ##
    ######################################################################
    history_df['f1_score'] = (
        2*(history_df['precision']*history_df['recall']) /
        (history_df['precision']+history_df['recall']+epsilon))
    history_df['val_f1_score'] = (
        2*(history_df['val_precision']*history_df['val_recall']) /
        (history_df['val_precision']+history_df['val_recall']+epsilon))

    history_df['f1_score_NEG'] = (
        2*(history_df['precision_NEG']*history_df['recall_NEG']) /
        (history_df['precision_NEG']+history_df['recall_NEG']+epsilon))
    history_df['val_f1_score_NEG'] = (
        2*(history_df['val_precision_NEG']*history_df['val_recall_NEG']) /
        (history_df['val_precision_NEG']+history_df['val_recall_NEG']+epsilon))

    history_df['f1_score_NEUT'] = (
        2*(history_df['precision_NEUT']*history_df['recall_NEUT']) /
        (history_df['precision_NEUT']+history_df['recall_NEUT']+epsilon))
    history_df['val_f1_score_NEUT'] = (
        2*(history_df['val_precision_NEUT']*history_df['val_recall_NEUT']) /
        (history_df['val_precision_NEUT']+history_df['val_recall_NEUT']+epsilon))

    history_df['f1_score_POS'] = (
        2*(history_df['precision_POS']*history_df['recall_POS']) /
        (history_df['precision_POS']+history_df['recall_POS']+epsilon))
    history_df['val_f1_score_POS'] = (
        2*(history_df['val_precision_POS']*history_df['val_recall_POS']) /
        (history_df['val_precision_POS']+history_df['val_recall_POS']+epsilon))
    ######################################################################


    ######################################################################
    ##                      Actual figure plotting                      ##
    ##                   ('class per metric' version)                   ##
    ######################################################################
    training_linestyle = (0, (3, 1, 1, 1))

    plt.ioff()
    fig1, axes = plt.subplots( figsize = fig1_size, nrows = 9, ncols = 1 )
    y = 1 ; line = plt.Line2D([0,1], [y, y], transform = fig1.transFigure, color = "black")
    fig1.add_artist(line)
    nrow = 0


    def draw_metric(ax_, col_name, title) :
        ax_.set_title(title)
        ax_.plot(history_df.index, history_df[ col_name ], label='training')
        ax_.plot(history_df.index, history_df[ 'val_' + col_name ], label='validation')

    def draw_class_metric(ax_, col_name, title, linestyle=None) :
        ax_.set_title(title)
        ax_.plot(history_df.index,history_df[ col_name ]
                 , linestyle=linestyle, label='weighted avg', color='black')
        ax_.plot(history_df.index,history_df[ col_name + '_NEG' ]
                 , linestyle=linestyle, label='NEGATIVE class', color='red')
        ax_.plot(history_df.index,history_df[ col_name + '_NEUT' ]
                 , linestyle=linestyle, label='NEUTRAL class', color='#c7be1c')
        ax_.plot(history_df.index,history_df[ col_name + '_POS' ]
                 , linestyle=linestyle, label='POSITIVE class', color='green')
        ax_.vlines(0, ymin=ax_.get_ylim()[0], ymax=ax_.get_ylim()[1],linewidth=.5)

    def set_common_yaxis(ax_0, ax_1) :
        # make sure 'ax_0' & 'ax_1' (dealing with the same performance metric) have common y-axis bounds
        ylim = (min(ax_0.get_ylim()[0], ax_1.get_ylim()[0]), max(ax_0.get_ylim()[1], ax_1.get_ylim()[1]))
        ax_0.set_ylim(ylim) ; ax_1.set_ylim(ylim)

    def gray_out_warming_up(ax_) :
        ax_.vlines(0, ymin=ax_.get_ylim()[0], ymax=ax_.get_ylim()[1],linewidth=.5)
        ax_.axvspan(xmin=min(history_df.index), xmax=0, ymin=.05, ymax=.95 , alpha=0.1, color='gray')

    def horizontal_separator(fig_, ax_, r_) :
        bbox = ax_.get_tightbbox(r_).transformed(fig_.transFigure.inverted())
        y=bbox.y0-.003
        line = plt.Line2D([0,1], [y, y], transform=fig_.transFigure, color="black")
        fig_.add_artist(line)

    ## LOSS ##
    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'loss', 'Training Loss')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Loss') ; ax.legend(loc="upper right", prop={'size': 8})
    gray_out_warming_up(ax)

    ## ACCURACY ##
    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'categorical_accuracy', 'Training Performance')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Accuracy')
    gray_out_warming_up(ax)

    ## CROSS-ENTROPY ##
    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'categorical_crossentropy', 'Training Performance')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy')
    gray_out_warming_up(ax)

    fig1.tight_layout() # DO IT HERE, TO ALLOW FOR CORRECT PLACEMENT OF HORIZONTAL SEPARATORS

    # SEPARATOR
    r = fig1.canvas.get_renderer()
    horizontal_separator(fig1, ax, r)

    ## PRECISION ##
    val_ax = axes[nrow] ; nrow += 1
    draw_class_metric(val_ax, 'val_precision', 'Training Performance (validation dataset)')
    ax = axes[nrow] ; nrow += 1
    draw_class_metric(ax, 'precision', 'Training Performance (training dataset)', linestyle=training_linestyle)
    set_common_yaxis( val_ax, ax )
    gray_out_warming_up( val_ax )
    gray_out_warming_up( ax )
    val_ax.set_xlabel('Epoch') ; val_ax.set_ylabel('Precision') ; val_ax.legend(loc="lower right", prop={'size': 7})
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Precision')

    # SEPARATOR
    horizontal_separator(fig1, ax, r)

    ## RECALL ##
    val_ax = axes[nrow] ; nrow += 1
    draw_class_metric(val_ax, 'val_recall', 'Training Performance (validation dataset)')
    ax = axes[nrow] ; nrow += 1
    draw_class_metric(ax, 'recall', 'Training Performance (training dataset)', linestyle=training_linestyle)
    set_common_yaxis( val_ax, ax )
    gray_out_warming_up( val_ax )
    gray_out_warming_up( ax )
    val_ax.set_xlabel('Epoch') ; val_ax.set_ylabel('Recall')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Recall')

    # SEPARATOR
    horizontal_separator(fig1, ax, r)

    ## F1 SCORE ##
    val_ax = axes[nrow] ; nrow += 1
    draw_class_metric(val_ax, 'val_f1_score', 'Training Performance (validation dataset)')
    ax = axes[nrow] ; nrow += 1
    draw_class_metric(ax, 'f1_score', 'Training Performance (training dataset)', linestyle=training_linestyle)
    set_common_yaxis( val_ax, ax )
    gray_out_warming_up( val_ax )
    gray_out_warming_up( ax )
    val_ax.set_xlabel('Epoch') ; val_ax.set_ylabel('F1 score')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('F1 score')

    # SEPARATOR
    y = -.005 ; line = plt.Line2D([0,1], [y, y], transform = fig1.transFigure, color = "black")
    fig1.add_artist(line)
    ######################################################################


    ######################################################################
    ##                      Actual figure plotting                      ##
    ##                   ('metric per class' version)                   ##
    ######################################################################

    fig2, axes = plt.subplots( figsize = fig2_size, nrows = 12, ncols = 1 )
    y = 1 ; line = plt.Line2D([0,1], [y, y], transform = fig2.transFigure, color = "black")
    fig2.add_artist(line)
    nrow = 0

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'precision', 'Training Performance')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Precision') ; ax.legend(loc="upper right", prop={'size': 8})
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'precision_NEG', 'Training Performance (NEGATIVE class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Precision')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'precision_NEUT', 'Training Performance (NEUTRAL class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Precision')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'precision_POS', 'Training Performance (POSITIVE class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Precision')
    gray_out_warming_up(ax)

    fig2.tight_layout() # DO IT HERE, TO ALLOW FOR CORRECT PLACEMENT OF HORIZONTAL SEPARATORS

    # SEPARATOR
    horizontal_separator(fig2, ax, r)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'recall', 'Training Performance')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Recall')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'recall_NEG', 'Training Performance (NEGATIVE class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Recall')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'recall_NEUT', 'Training Performance (NEUTRAL class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Recall')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'recall_POS', 'Training Performance (POSITIVE class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('Recall')
    gray_out_warming_up(ax)

    # SEPARATOR
    horizontal_separator(fig2, ax, r)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'f1_score', 'Training Performance')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('F1 score')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'f1_score_NEG', 'Training Performance (NEGATIVE class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('F1 score')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'f1_score_NEUT', 'Training Performance (NEUTRAL class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('F1 score')
    gray_out_warming_up(ax)

    ax = axes[nrow] ; nrow += 1
    draw_metric(ax, 'f1_score_POS', 'Training Performance (POSITIVE class)')
    ax.set_xlabel('Epoch') ; ax.set_ylabel('F1 score')
    gray_out_warming_up(ax)

    # SEPARATOR
    y = -.005 ; line = plt.Line2D([0,1], [y, y], transform = fig2.transFigure, color = "black")
    fig2.add_artist(line)
    ######################################################################


    return fig1, fig2




















#/////////////////////////////////////////////////////////////////////////////////////























































