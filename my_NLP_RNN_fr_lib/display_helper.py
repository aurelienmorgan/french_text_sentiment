import os, re, random
import pandas as pd, numpy as np
import copy

import base64

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.ticker import StrMethodFormatter

from IPython.display import display_html


#/////////////////////////////////////////////////////////////////////////////////////


table_header_pattern = re.compile('<th( class="col_heading level\d+ col\d+" ){0,1}>(.*)</th>', re.MULTILINE)

def format_vertical_headers(
    pd_dataframe: pd.DataFrame
    , header_height_px = 180, left_margin_px = 5, bottom_margin_px = 5, min_column_width_px = 45
    , greyed_out_colnames = None
) -> None :
    """
    Displays a Pandas DataFrame with vertical headers
    in a IPython cell output
    """

    table_id = str(random.randint(1, 10**10))

    html_ = pd_dataframe.to_html(index=False) if isinstance(pd_dataframe, pd.DataFrame) \
            else pd_dataframe.hide_index().render().replace('</th>', '</th>\n') if isinstance(pd_dataframe, pd.io.formats.style.Styler) \
            else None
    if html_ is None : raise TypeError("input of type '" + type(pd_dataframe) + "' not handled.")

    if greyed_out_colnames :
        def convert_func(matchobj):
            column_header_raw_text =  matchobj.group(2) #column header text
            column_header_node_text = matchobj.group(0)
            #print(column_header_node_text + " - " + column_header_raw_text)
            
            return \
                column_header_node_text.replace(column_header_raw_text
                                                , '<p style="color: darkgray;">'+column_header_raw_text+'</p>') \
                if column_header_raw_text in greyed_out_colnames \
                else column_header_node_text

        html_ = table_header_pattern.sub(convert_func, html_)
        #print(html_)

    html_ = \
        """
        <!DOCTYPE html><html><head>
        <style>
        .rotate""" + table_id + """ div {
            position:   absolute;
            white-space:  nowrap;
            left:       """ + str(left_margin_px) + """px;                        /* rotated header bottom margin */
            top:        """ + str(header_height_px + bottom_margin_px) + """px;   /* height + rotated header left margin */
            /* attempt at right-alligning, short on large columns (and/or large 'min_column_width_px') => */
            transform: translateY(100%);
            #background-color: yellow;
        }

        .rotate""" + table_id + """ {
            transform:          rotate(-90deg);
            transform-origin:          0% 100%;
            height:                      """ + str(header_height_px) + """px;
            width: """ + str(min_column_width_px) + """px;
            #background-color: red;
        }
        </style></head>
        <body><center>
        """ + \
        table_header_pattern.sub('<th class="rotate' + table_id + '"><div>\\2</div></th>'
                                 , html_) + \
        """</center></body></html>"""
    #print( html_ )

    display_html( html_, raw=True )



#/////////////////////////////////////////////////////////////////////////////////////


def dataframe_pretty_print_center(df) :
    """
    Prettifies a dataframe and displays it centered in an IPython cell

    Parameters :
        df (pandas.DataFrame) : the dataframe to print
    """

    html_ = \
        """
        <!DOCTYPE html><html><head>
        <style>
        div.centered_table_container {
            text-align: center;
        }
        table {
            display: inline-block;
        }
        </style></head>
        <body>
        <div class="centered_table_container">""" + \
        df.style.hide_index(
        ).set_table_styles(
            [dict(selector="th", props=[('text-align','center')])]
        ).set_properties(**{'width':'20em', 'text-align':'center'}
        ).render() + \
        '</div></body></html>'
    #print( html_ )

    display_html(
        html_
        , raw=True
    )


#/////////////////////////////////////////////////////////////////////////////////////


two_columns_cell_css = \
"""
* {
    box-sizing: border-box;
}

.left_column {
    float: left;
    padding: 10px;
    #padding-left: 80px;
}

.right_column {
    float:right;
    text-align; right;
    padding: 10px;
    #padding-left: 80px;
    #background: yellow;
}

/* Clearfix (clear floats) */
.row::after {
    content: "";
    clear: both;
    display: table;
}
"""

two_columns_cell_body = \
"""
<!DOCTYPE html>
<html>
<head>
<style>
#two_columns_cell_css
</style>
</head>
<body>

<h2>#title</h2>

<div class="row" style="width: 100%;">
  <div class="left_column" style="width: #left_width;">
      <center>
          #left_content
      </center>
  </div>
  <div class="right_column" style="width: #right_width;">
      <center>
          #right_content
      </center>
  </div>
</div>
</body>
</html>
"""

def two_columns_display(left_content, right_content, left_ratio = .5, header = '') :
    """
    Inspired from 'https://stackoverflow.com/questions/45286696/#45287049'
    (how-to-display-dataframe-next-to-plot-in-jupyter-notebook)

    Displays IPython cell content as a 2-columns element

    Arguments:
        left_content (string) : html text of the left column
        right_content (string) : html text of the right column
        header (string) : cell header
        left_ratio (float) : proportion of width alloted to the left column
    """
    html = two_columns_cell_body \
        .replace('#title', header) \
        .replace('#two_columns_cell_css'
                 , two_columns_cell_css
                ) \
        .replace(
             '#left_width'
             , "{:2.2f}%".format(100*left_ratio)
         ).replace(
             '#right_width'
             , "{:2.2f}%".format(100*(1-left_ratio))
         ) \
        .replace('#left_content', left_content) \
        .replace('#right_content', right_content)
    #print( html )

    display_html(html, raw=True)


#/////////////////////////////////////////////////////////////////////////////////////


def two_columns_display__classes_distribution(
    dataset_labels: pd.core.series.Series
) -> None :
    """
    Displays (in an IPython cell output) 2 columns :
        - on the left, a table of classes distribution
          (incl. percentage column)
        - on the right, a barchart plot of classes distribution

    Parameters :
        - dataset_labels (pd.core.series.Series) :
            a collection of dataset records labels.
    """

    df = pd.DataFrame(dataset_labels.copy())
    df.reset_index(level=0, inplace=True) # index-to-column
    df = df.groupby('sentiment').count() \
        .rename(columns = {'index':'count'})
    df[ 'percentage' ] = (df / len( dataset_labels ) * 100)['count']

    df_html = df.reset_index(level=0).style.hide_index() \
        .set_properties(**{'width':'20em', 'text-align':'center'}) \
        .format({"count": "{:,}", "percentage": "{0:.2f}%"}) \
        .set_table_styles([dict(selector="th", props=[('text-align', 'center')])]).render()

    if not os.path.isdir('tmp') : os.makedirs('tmp')
    title = "Imbalanced_dataset_downsampled"
    fig_url = os.path.join('tmp', title+".jpg")

    plt.ioff() ; fig = plt.figure() ; ax = plt.gca();
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    df['count'].plot( kind='bar', figsize=(6,3), ax = ax) ; #plt.show()
    plt.savefig(fig_url, bbox_inches='tight') ; plt.close(fig)

    # use base64 encoding to circumvent image (external file) caching (webbrowser)
    with open(fig_url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    encoded_string = "data:image/jpeg;charset=utf-8;base64, " + encoded_string.decode()
    os.remove(fig_url)

    two_columns_display(df_html, '<img src="' + encoded_string + '" />', .5)


#/////////////////////////////////////////////////////////////////////////////////////


def classification_report_pretty_html(
    report_dict: dict
) -> str :
    """
    Pretty-formatted html representation.

    Parameters :
        - report_dict (dict) :
            a classification report as generated by the
            `sklearn.metrics.classification_report` method.

    Results :
        - report_html (str) :
            an html string.
    """

    report_df = pd.DataFrame(copy.deepcopy(report_dict)).transpose() 
    report_df.loc['accuracy', ['precision', 'recall']] = np.nan
    report_df.loc['accuracy', 'support'] = report_df.loc['macro avg', 'support']

    def css_border(x) :
        return ["border-top: 1px solid black" if (i==(x.shape[0]-classes_count-1))
                else "border: 0px" for i, row in enumerate(x)]

    classes_count = report_df.shape[0]-3
    report_html = (
        pd.concat([
            report_df.iloc[:classes_count]
            , pd.DataFrame({'f1-score':[np.nan]}, index=['&nbsp;']) # insert empty row
            , report_df.iloc[classes_count:]], axis=0, sort=False
        ).reset_index(level=0).rename(columns={'index': ''})        # integrate 'index' column
                                                                    # for css border to go all along the row
        .style.format(
            {
                'precision': lambda x: '' if pd.isnull(x) else '{:,.2f}'.format(x)
                , 'recall': lambda x: '' if pd.isnull(x) else '{:,.2f}'.format(x)
                , 'f1-score': lambda x: '' if pd.isnull(x) else '{:,.2f}'.format(x)
                , 'support': lambda x: '' if pd.isnull(x) else '{:,g}'.format(x)
            }
        ).apply(css_border, axis=0).hide_index().render()
    )

    return report_html


#/////////////////////////////////////////////////////////////////////////////////////

































