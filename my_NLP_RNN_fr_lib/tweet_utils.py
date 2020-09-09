import numpy as np
import pandas as pd

import re
from lxml import html

import os, gc
import time
import zipfile
import subprocess

from .gzip_utils import download_progress

import matplotlib.pyplot as plt

from matplotlib import ticker

from IPython.display import display_html


#/////////////////////////////////////////////////////////////////////////////////////


TWEET_MAX_CHAR_COUNT = 280


#/////////////////////////////////////////////////////////////////////////////////////


def download_tweet_fr() :
    root_dir = os.path.join( 'data', 'tweets_fr' )
    if not os.path.isdir(root_dir) : os.makedirs( root_dir )
    root_url = "https://raw.githubusercontent.com/charlesmalafosse/" + \
        "open-dataset-for-sentiment-analysis/master/"
    url = root_url + "betsentiment-FR-tweets-sentiment-teams.zip"
    filefullname = os.path.join( root_dir, 'betsentiment-FR-tweets-sentiment-teams.csv')
    zipfullname = filefullname.replace('.csv', '.zip')
    if not os.path.isfile(filefullname) :
        download_progress(url , zipfullname)
        directory_to_extract_to = os.path.abspath(os.path.join(zipfullname, os.pardir))
        with zipfile.ZipFile(zipfullname, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    filefullname = os.path.join( root_dir, 'betsentiment-FR-tweets-sentiment-worldcup.csv')
    if not os.path.isfile(filefullname) :
        url =  root_url + "betsentiment-FR-tweets-sentiment-worldcup-split.zip.002"
        zip2fullname = filefullname.replace('.csv', '-split.zip.002')
        download_progress(url , zipfullname)
        url =  root_url + "betsentiment-FR-tweets-sentiment-worldcup-split.zip.001"
        zip1fullname = filefullname.replace('.csv', '-split.zip.001')
        download_progress(url , zipfullname)
        directory_to_extract_to = os.path.abspath(os.path.join(zip1fullname, os.pardir))
        unzip_command = '"%ProgramFiles%\WinRAR\winrar.exe"' +  ' -y -ibck -im x ' + \
            '"' + os.path.abspath(zip1fullname) + '"' +  ' *.* . '
        result = subprocess.run(unzip_command, shell = True, cwd = directory_to_extract_to)
        ERR_MESSAGE = '''
        multi-part decompression failed (returncode: ''' + str(result.returncode) + ''')
        aren't you running on Windows OS with WinRar installed ?
        '''
        assert result.returncode == 0, ERR_MESSAGE


#/////////////////////////////////////////////////////////////////////////////////////


def load_tweet_fr() :
    tic = time.perf_counter()

    root_path = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__))
            , "..", 'data', 'tweets_fr'
        )
    )
    #print(root_path)

    tweets = pd.read_csv(
        os.path.join( root_path, 'betsentiment-FR-tweets-sentiment-teams.csv' )
        , encoding='latin1' )
    tweets['src'] = 'teams'
    tweets2 = pd.read_csv(
        os.path.join( root_path, 'betsentiment-FR-tweets-sentiment-worldcup.csv' )
        , encoding='latin1' )
    tweets2['src'] = 'worldcup'
    tweets = pd.concat( [ tweets, tweets2 ] )
    del tweets2
    gc.collect()

    tweets.drop(['tweet_date_created', 'tweet_id', 'language', 'sentiment_score']
                , axis=1, inplace=True)
    tweets[ 'sentiment' ] = np.where(
        tweets[ 'sentiment' ] == 'MIXED', 'NEUTRAL', tweets[ 'sentiment' ] )

    toc = time.perf_counter()
    print(f"Loaded the dataset in {toc - tic:0.4f} seconds")

    return tweets


#/////////////////////////////////////////////////////////////////////////////////////


def load_review_fr() :
    tic = time.perf_counter()

    root_path = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__))
            , "..", 'data', 'web_scraping'
        )
    )
    #print(root_path)

    # bars & restaurants
    file_path = os.path.join(
            root_path, 'bars_n_restaurants.csv'
    )
    #print(file_path)
    reviews = pd.read_csv( file_path, encoding = 'utf-8' )
    reviews.columns = [ 'rating', 'comment' ]
    reviews[ 'type' ] = 'bars_n_restaurants'

    # movies
    file_path = os.path.join(
            root_path, 'movies.csv'
    )
    #print(file_path)
    tmp = pd.read_csv( file_path, encoding = 'utf-8' )
    tmp.columns = [ 'rating', 'comment' ]
    reviews = \
        pd.concat( [ reviews, tmp ], axis = 0, sort = False )
    reviews[[ 'type' ]] = reviews[[ 'type' ]].fillna(value='movies')
    del tmp
    gc.collect()

    # appliances & electronics
    file_path = os.path.join(
            root_path, 'appliances_n_electronics.csv'
    )
    tmp = pd.read_csv( file_path, encoding = 'utf-8' )
    tmp.columns = [ 'rating', 'comment' ]
    reviews = \
        pd.concat( [ reviews, tmp ], axis = 0, sort = False )
    reviews[[ 'type' ]] = reviews[[ 'type' ]].fillna(value='appliances_n_electronics')
    del tmp
    gc.collect()
    # assimilate the "0" grades to "1" grades, in order
    # to have the same grading rande as for the other data sources : [1-5] 
    reviews.loc[ reviews['rating'] == 0, 'rating' ] = 1

    reviews.dropna(subset = ['comment'])
    reviews[["comment"]] = reviews[["comment"]].astype(str)
    reviews.reset_index(drop=True, inplace=True)

    toc = time.perf_counter()
    print(f"Loaded the dataset in {toc - tic:0.4f} seconds")

    return reviews


#/////////////////////////////////////////////////////////////////////////////////////


def strip_html(s):
    """
    REMOVE HTML TAGS
    """
    return ('' if (s is None) or (s == '') else str(html.fromstring(s).text_content()))

special_char_map = \
    {ord(''):'oe', ord(''):"'", ord(''):" ", ord(''):''
     , ord(''):'', ord(''):"'", ord(''):"'", ord('"'):""}
def str_clean( s ) :
    """
    REMOVE SPECIAL CHARACTERS (such as 'œ')
    """
    result = strip_html( ''.join( [value.translate( special_char_map ) for value in s] ) )
    result = re.sub( '&nbsp;', ' ', result )
    result = re.sub( '’', "'", result )
    # remove duplicate '!'
    result = re.sub( '!+', '!', result )
    # add missing space after '!' (except when followed by ')')
    result = re.sub( '!(?!' + re.escape(")") + '| )', '! ', result )
    # add missing space after ')' (except when followed by '.')
    result = re.sub( re.escape(")") + '(?!' + re.escape(".") + '| )', ') ', result )
    # shorten excessively long '...'
    result = re.sub(
        re.escape(".") + re.escape(".") + re.escape(".") + '(' + re.escape(".") + '+)'
        , '...', result )
    # add missing space after '.' (except when followed by '.')
    result = re.sub( re.escape(".") + '(?!' + re.escape(".") + '| |$)', '. ', result )
    # remove single quotes (but NOT apostrophes)
    result = re.sub( '( |^)' + re.escape("'") + '+', " ", result )
    result = re.sub( '(' + re.escape("'") + ')( |' + re.escape(".") + '|$)', r"\2", result )
    return result

retweet_pattern = re.compile(r'(?m)\n?^Retweet.*\n?')
usernames_pattern = re.compile('@[^\s]+')
urls_pattern = re.compile(r"http\S+")

def clean_tweet_fr(tweets_df, col_name = 'tweet_text') :

    # remove 'Retweet' trailing line =>
    tweets_df[ col_name ] = tweets_df[ col_name ].map(
        lambda comment: retweet_pattern.sub('', str(comment))
    )

    # remove referenced user names from tweets =>
    tweets_df[ col_name ] = tweets_df[ col_name ].map(
        lambda comment: usernames_pattern.sub('',str(comment))
    )

    # remove urls, special characters and html tags from tweets =>
    tweets_df[ col_name ] = tweets_df[ col_name ].map(
        lambda comment: str_clean(urls_pattern.sub(
            "", str(comment).strip()
        ))
    )

    # convert hashtag within tweets =>
    tweets_df[ col_name ] = \
        hashtag_converter_tweet_fr(tweets_df[ col_name ])

    # remove duplicated tweets =>
    tweets_df.drop(tweets_df[tweets_df.duplicated(col_name)].index, inplace=True)
    tweets_df.reset_index(drop=True, inplace=True)

    return tweets_df


#/////////////////////////////////////////////////////////////////////////////////////


# split by words starting by one upper case
# note the importance of the surrounding paranthesis
# tokens that do not match are not split but kept
# (sequence of words with no upper case for instance)
upper_case_initial_pattern = re.compile('([A-Z][a-zà-ÿ]+)')

two_or_more_spaces_pattern = re.compile(' {2,}')
leading_trailing_space_pattern = re.compile('(?m)(^ )|( $)')
# extra spaces around hour separator
hour_separator_leading_space_pattern = re.compile('(?i)(h|:) (?=\d)')
hour_separator_trailing_space_pattern = re.compile('(?i)(?<=\d) (h|:)')
# extra space in "1 er" or "2 ème", etc.
rank_abbr_space_pattern = re.compile('(?<=\d) (er|[eéè]me)')

def hastag_converter(tweet_text) :
    list_ = \
        upper_case_initial_pattern.split(
           # first, below, taking care of digits
           # (which could be surrounded by characters)
           ' ' .join(re.split('(\d+)'
                              , tweet_text)
                    )
        )
    #print(list_)
    result = ' ' .join( list_ ).replace('#', '')

    result = \
        leading_trailing_space_pattern.sub(
            '', two_or_more_spaces_pattern.sub(' ', result)
        )
    result = \
        hour_separator_leading_space_pattern.sub(
            r"\1", hour_separator_trailing_space_pattern.sub(r"\1", result)
        )
    result = \
        rank_abbr_space_pattern.sub(r"\1", result)

    return \
        result.strip() \
            .replace(' .', '.').replace(' ,', ',').replace(' :', ':') \
            .replace(' - ', '-').replace(r' / ', r'/') \
            .replace('( ', '(').replace(' )', ')')

def hashtag_converter_tweet_fr(tweets_df_column) :
    return tweets_df_column.map( hastag_converter )


# unit_test
str_ = '#tousAvecLesBleus #AllezLesBleus #Mettre3-0 #viveLePSG à 17h30 JKL Algériens U victor ; ' + \
    'MNOP AB étienne #Titi127Tata3titi Didier BBCNews. AhAh!'
assert hastag_converter(str_) == \
    'tous Avec Les Bleus Allez Les Bleus Mettre 3-0 vive Le PSG à 17h30 JKL Algériens U victor ; MNOP AB étienne Titi 127 Tata 3 titi Didier BBC News. Ah Ah !' \
    , 'hastag_converter flawed'


#/////////////////////////////////////////////////////////////////////////////////////


from html.entities import codepoint2name
html_escape_table = {k: '&{};'.format(v) for k, v in codepoint2name.items()}

def html_escape(string: str) -> str :
    return string.translate(html_escape_table)


#/////////////////////////////////////////////////////////////////////////////////////


def my_log_formatter_fun(y, pos):
    """
    inspired from 'https://stackoverflow.com/questions/21920233#33213196'
    (matplotlib-log-scale-tick-label-number-formatting)
    
    to be used as a "matplotlib.ticker.FuncFormatter"
    for axis label formatting in log scale.
    displays numbers larger than 0 as integers (no floating point),
    as floats otherwize.
    
    Arguments:
        - a tick value ``x``
        - a position ``pos``
    
    result : a string containing the corresponding tick label.
    """

    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:,.{:1d}f}}'.format(decimalplaces)

    # Return the formatted tick label
    return formatstring.format(y)

def plt_hist_bin_width( data, binwidth, ax = None, ylog = False, rotate_text = True ) :
    """
    inspired from 'https://stackoverflow.com/questions/5328556'
    (histogram-matplotlib)
    and from  'https://stackoverflow.com/questions/6986986#12176344'
    (binwidth ; bin-size-in-matplotlib-histogram)
    
    plots an histogram where the bars are "centered" on the upper bound.
    For instance, the bar at position "x=binwidth" represent
    the value for '0<x<=binwidth'.
    
    parameters :
        - 'data' : array_like
                   Input data. The histogram is computed over the flattened array.
        - 'binwidth' : int
        - 'ax' : an object of class 'matplotlib.axes.Axes'
                 to be populated with chart visual
                 If 'not defined, a new one is created
        - 'ylog' : boolean
                   whete=her or not the y-axis is to be set to 'log' scale
        - 'rotate_text' : int
                          angle by which the bars labels shall be rotated
    
    result : an object of class 'matplotlib.axes.Axes'
    """

    # hist, bins = np.histogram( data, bins = 10 ) <<== if we rather simply specify how many 'bins' we want

    # "binwidth+1" to shift the window to include the upper bound
    # (& exclude lower bound) for each bin
    bins = np.arange(min(data)+1, max(data) + binwidth, binwidth)
    #print( bins )
    hist, _ = np.histogram( data, bins = bins )

    width = .7 * ( bins[1] - bins[0] )
    center = ( ( bins[:-1] + bins[1:] + binwidth - 1 ) / 2 ).astype(np.int64)
    #print( center )

    if ax is None :
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 1, 1)
    ax.bar( center, hist, align = 'center', width = width )
    ax.set_xticks(center)
    ax.set_xticklabels(center)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    if ylog :
        ax.set_yscale('log')
        ax.set_ylim(ax.get_ylim()[0], 1000000)
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(my_log_formatter_fun))

    rects = ax.patches
    if rotate_text : ha='left' ; rotation = 45
    else : rotation = 0 ; ha='center'
    for rect, label in zip(rects, hist):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, '{:,}'.format(label)
                , ha=ha, va='bottom'
                , rotation=rotation)

    return ax


#/////////////////////////////////////////////////////////////////////////////////////






















