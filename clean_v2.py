# COMP90051 Project 1 source code
# For Team 192

import spacy, re
from spacymoji import Emoji # credit to: https://github.com/ines/spacymoji for the spacy extension
import ext_spacy

def get_nlp_v2():
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    emoji = Emoji(nlp)
    nlp.add_pipe(emoji, first=True)

    # bug with spacy https://github.com/explosion/spaCy/issues/1574
    for word in nlp.Defaults.stop_words.difference(ext_spacy.stop_words_modified):
        nlp.vocab[word].is_stop = False

    return nlp

def tweet_clean_v2(txt):
    txt = re.sub('[’]', "'", txt)
    txt = re.sub('[()]', '', txt)
    txt = re.sub('@handle', '', txt)
    return txt

def tweet_spacy_v2(doc):
    def spacy_token_filter(t):
        if len(t) >= 2 and t.text in ext_spacy.emoticons:
            return t.text + ' '
        
        conds = [not t.is_punct, not t.is_stop,
                 not t.pos == 'X' or t.is_alpha]
        
        if all(conds):
            out = t.lemma_
            
            if t.like_num:
                out = '<NUM>'
            elif t.like_url:
                matches = re.match(r'https?:\/\/(?:www[.])?((?!w+[.]*)[a-zA-Z0-9]+[.][a-zA-Z]+){1}(?:\/.*)?', out) # e.g. https://www.google.com/blah -- we only want google.com section
                if matches is not None:
                    out = matches[1]
            
            out = re.sub(r'([0-9]{1,2}:[0-9]{1,2})(AM|PM|am|pm)?', r'<TIME> \2', out) # time formatting
            out = out.rstrip('.') # changes something like Mr. to Mr
            out = re.sub(r'[-]', '', out) # no dashes
            return out + ' '
        
        return t.whitespace_
    
    # lemmatization with filtering
    ret = ''.join([spacy_token_filter(token) for token in doc])
    return ret.strip().lower()