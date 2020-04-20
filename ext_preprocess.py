# COMP90051 Project 1 source code
# For Team 192

from IPython.display import display, clear_output
import time
from spacy.matcher import Matcher

import pandas as pd
import swifter
from sklearn.model_selection import train_test_split

import clean_v1, clean_v2, clean_v3

def load_ml_files(train_f, test_f):
    # Load train and test CSV comma-delimited files
    with open(train_f, 'r') as f:
        data = [x.split('\t') for x in f.read().splitlines()]
        df_train = pd.DataFrame(data, columns=['label', 'tweet'])
        df_train = df_train.astype({'label': int, 'tweet': str})

    with open(test_f, 'r') as f:
        df_test = pd.DataFrame(f.read().splitlines(), columns=['tweet'])

    return df_train, df_test

'''
def spacy_preserve_hashtags(doc):
    # CREDIT TO: https://stackoverflow.com/a/53944339 for the code
    # treat hashtags like #Obama2020 together (not # and Obama2020 separate)
    matcher = Matcher(nlp.vocab)
    matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])

    matches = matcher(doc)
    hashtags = []
    for match_id, start, end in matches:
        hashtags.append(doc[start:end])

    for span in hashtags:
        span.merge()
'''

def wrap_progress(f, iter, total, UPDATE_INTERVAL=500, EXTRA_TEXT=''):
    processed = 0

    for item in iter:
        f(item)
        processed += 1

        if processed % UPDATE_INTERVAL == 0:
            clear_output(wait=True)
            display("Processing: %s... %d / %d (%.4f)" %
                        (EXTRA_TEXT, processed, total, float(processed)/total))


SPACY_FUNCS = {
    'cleaned': [clean_v1.get_nlp_v1, clean_v1.tweet_clean_v1, clean_v1.tweet_spacy_v1],
    'cleaned_v2': [clean_v2.get_nlp_v2, clean_v2.tweet_clean_v2, clean_v2.tweet_spacy_v2],
    'cleaned_v3': [clean_v3.get_nlp_v3, clean_v3.tweet_clean_v3, clean_v3.tweet_spacy_v3],
}

def clean_instances(df_X, col_name, batch_size=100):
    new_text = []
    nlp = SPACY_FUNCS[col_name][0]()

    def process_doc(doc):
        if doc.is_parsed:
            new_text.append(SPACY_FUNCS[col_name][2](doc))

    df = df_X.copy()

    # before parsing with spacy
    df[col_name] = df['tweet'].swifter.apply(SPACY_FUNCS[col_name][1])

    # use spacy's much faster pipe processor (allows multiprocessing)
    new_text = []
    wrap_progress(process_doc, \
        nlp.pipe(df[col_name].values, batch_size=batch_size), \
        df.shape[0],
        EXTRA_TEXT="Cleaning using approach %s" % (col_name))
    
    df[col_name] = new_text
    return df
