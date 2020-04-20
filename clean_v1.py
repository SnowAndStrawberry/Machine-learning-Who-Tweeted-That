# COMP90051 Project 1 source code
# For Team 192

import spacy, re

def get_nlp_v1():
    return spacy.load('en_core_web_sm', disable=['ner'])  

def tweet_clean_v1(twt):
    ret = twt
    # remove @handle's
    ret = ret.replace('@handle', '')
    # remove URL links
    ret = re.sub(r'https?://?\S+', '', ret)

    return ret

def tweet_spacy_v1(doc):
    def spacy_token_filter(token):
        if not token.is_stop and not token.is_punct and not \
        (token.pos_ == 'X' and not token.is_alpha):
            return token.lemma_ + ' '
    
        return token.whitespace_
    
    # TODO: preserve hashtags?
    # spacy_preserve_hashtags(doc)
    
    # lemmatization with filtering
    ret = ''.join([spacy_token_filter(token) for token in doc])
    return ret.strip().lower()