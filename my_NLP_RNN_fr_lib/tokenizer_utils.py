import re

my_pattern_str = (re.escape("(") +
    "|" + re.escape(")") +
    "|" + re.escape(".") + re.escape(".") + re.escape(".") +
    "|" + re.escape(".") +
    '|\d+(?=(?i:kg))(?!(?i:kg)[A-z]+)' # abbreviation ; 'kilogrammes'
    '|(?i:kg)(?![A-z]+)'
    '|\d+(?=(?i:kgs))(?!(?i:kgs)[A-z]+)' # abbreviation ; 'kilogrammes'
    '|(?i:kgs)(?![A-z]+)'
    '|\d+(?=gr)(?!gr[A-z]+)' # abbreviation ; 'grammes'
    '|gr(?![A-z]+)'
    '|\d+(?=g)(?!g[A-z]+)' # abbreviation ; 'grammes'
    '|g(?!\w+)'
    '|\d+(?=ml)(?!ml[A-z]+)' # abbreviation ; 'ml'
    '|ml(?!\w+)'
    '|\d+(?=(?i:hz))(?!(?i:hz)[A-z]+)' # abbreviation ; 'hertz'
    '|(?i:hz)(?![A-z]+)'
    '|\d+(?=jrs)(?!jrs[A-z]+)' # abbreviation ; 'jours'
    '|jrs(?!\w+)'
    '|\d+(?=jr)(?!jr[A-z]+)' # abbreviation ; 'jours'
    '|jr(?!\w+)'
    '|\d+(?=j)(?!j[A-z]+)' # abbreviation ; 'jours'
    '|j(?!\w+)'
    '|\d+(?=(?i:h))(?!(?i:h)[A-z]+)' # abbreviation ; 'heure'
    '|(?<=(?i:h))\d+'
    '|(?<=\d)(?i:h)(?![A-z]+)'
    '|\d+(?=(?i:hr))(?!(?i:hr)[A-z]+)' # abbreviation ; 'heure'
    '|(?<=(?i:hr))\d+'
    '|(?<=\d)(?i:hr)(?![A-z]+)'
    '|\d+(?=min)(?!min[A-z]+)' # abbreviation ; 'minutes'
    '|min(?!\w+)'
    '|\d+(?=km)(?!km[A-z]+)' # abbreviation ; 'kilomètres'
    '|(?<=\d)km(?![A-z]+)'
    '|\d+(?=m)(?!m[A-z]+)' # abbreviation ; 'mètres'
    '|(?<=\d)m(?![A-z]+)'
    '|(?<=m)\d+'
    '|\d+(?=cm)(?!cm[A-z]+)' # abbreviation ; 'cm'
    '|cm(?!\w+)'
    '|\d+(?=ans)(?!ans[A-z]+)' # 'ans'
    '|ans(?!\w+)'
    '|\d+(?=(?i:euros))(?!(?i:euros)[A-z]+)' # 'euros'
    '|(?<=\d)(?i:euros)(?![A-z]+)'
    '|(?<=(?i:euros))\d+'
    '|\d+(?=(?i:euro))(?!(?i:euro)[A-z]+)' # 'euros'
    '|(?<=\d)(?i:euro)(?![A-z]+)'
    '|(?<=(?i:euro))\d+'
    '|\d+(?=(?i:eur))(?!(?i:eur)[A-z]+)' # 'euros'
    '|(?<=\d)(?i:eur)(?![A-z]+)'
    '|(?<=(?i:eur))\d+'
    '|\d+(?=(?i:€))(?!(?i:€)[A-z]+)' # 'euros'
    '|(?<=\d)(?i:€)(?![A-z]+)'
    '|(?<=(?i:€))\d+'
    '|\d+(?=ième)(?!ième[A-z]+)' # abbreviation ; 'n-ième'
    '|ième(?!\w+)'
    '|\d+(?=ème)(?!ème[A-z]+)' # abbreviation ; 'n-ième'
    '|ème(?!\w+)'
    '|\d+(?=eme)(?!eme[A-z]+)' # abbreviation ; 'n-ième'
    '|eme(?!\w+)'
    '|\d+(?=e)(?!e[A-z]+)' # abbreviation ; 'n-ième'
    '|e(?!\w+)'
    '|\d+(?=è)(?!è[A-z]+)' # abbreviation ; 'n-ième'
    '|è(?!\w+)'
    "|" + r"\\" + "|/|!|,|;|[\w]+|[\S]"
)
my_pattern = re.compile(my_pattern_str)

def tokenize_fr(text) :
    return my_pattern.findall(text)

def tokenize_tweet_fr(df, col_name='tweet_text') :
    """
    Returns :
        - a pandas.core.series.Series. Each entry consists of
          a list of tokens per input sentence
    """
    return df[col_name].map(lambda comment: tokenize_fr(comment))


# unit_test
s = '($60 13H22 14h! 16-18h 19h/20h titi toto tata' + \
    ' 250gr 200gr. 300g 13h 17H15 70km 33m 11m93 8Euros 14euros90 230ml' + \
    ' 250ème 10ans 16eme 120Hz' + \
    " de l'appeler 'Calimero'."
tokenized_s = tokenize_fr(s)
assert len(tokenized_s) == 63, 'pattern tokenizer flawed'


#////////////////////////////////////////////////////////////////////////////////////////////////////


wordcloud_exclude = \
set(['sur', 'car', 'par', 'les'
     , 'cet', 'cette', 'des', 'donc'
     , 'que', 'qui', 'pour', 'dans', 'une'
     , 'est', 'était', 'fait'
     , 'mon', 'moi', 'son', 'lui', 'nous', 'vous', 'notre'
     , 'avec', 'tout', 'faut', 'fait', 'faire', 'même', 'cela', 'car', 'être', 'sont'
    , 'mais', 'peu', 'pas', 'très', '...'])