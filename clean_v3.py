# COMP90051 Project 1 source code
# For Team 192

import spacy, re, html
from spacymoji import Emoji # credit to: https://github.com/ines/spacymoji for the spacy extension
import ext_spacy
import string 

def get_nlp_v3():
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    emoji = Emoji(nlp)
    nlp.add_pipe(emoji, first=True)

    # bug with spacy https://github.com/explosion/spaCy/issues/1574
    for word in nlp.Defaults.stop_words.difference(ext_spacy.stop_words_modified):
        nlp.vocab[word].is_stop = False

    return nlp

def tweet_clean_v3(txt):
    txt = html.unescape(txt) # remove html entites
    txt = re.sub('’', "'", txt)
    txt = re.sub('@handle', '', txt) 
    txt = re.sub('=', ":", txt) # normalising emoticons (shouldn't conflict with URLs)

    return txt

def tweet_spacy_v3(doc):
    def spacy_token_filter(t):
        if len(t) >= 2 and t.text in ext_spacy.emoticons:
            return t.text + ' '
        
        remove_puncts = True
        avoid_token = [t.is_punct, t.is_stop, t.pos == 'X' and (not t.is_alpha)]
        
        if not any(avoid_token):
            out = t.lemma_.lower()
            
            if t.like_num:
                out = '<LIKE_NUM>'
                remove_puncts = False
            elif t.pos_ == 'NUM' and ':' in t.text:
                out = '<TIME>'
                remove_puncts = False
            elif t.like_url:
                matches = re.match(r'(?:https?:\/\/)?(?:www[.])?((?!w+[.]*)[a-zA-Z0-9]+[.][a-zA-Z]+){1}(?:\/.*)?', out) # e.g. https://www.google.com/blah -- we only want google.com section
                if matches is not None:
                    out = matches[1]
                    remove_puncts = False
            
            out = re.sub(r'([0-9]{1,2}:[0-9]{1,2})(AM|PM|am|pm)?', r'<TIME> \2', out) # time formatting
            out = out.rstrip('.') # changes something like Mr. to Mr
            out = re.sub(r'[-]', '', out) # no dashes

            #out = re.sub('[()]', '', out)

            if remove_puncts:
                # credit to https://stackoverflow.com/a/43934982 for the snippet
                # replaces punctuation with whitespace (quicker than calling str.replace)
                out = out.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) 

            # 3 or more consecutive letters at the end (don't think it happens in English)
            # only at start and end, easy to do (harder to figure out inside unless we use spelling correction)
            if len(out) > 2:
                i = 1
                c_last = out[0]
                for c in out[1:]:
                    if c == c_last:
                        i += 1
                    else:
                        break
                        
                if i > 2:
                    out = out[i-2:]

            if len(out) > 2:
                i = 1
                c_last = out[-1]
                for c in (out[:-1])[::-1]:
                    if c == c_last:
                        i += 1
                    else:
                        break
                        
                if i > 2:
                    out = (out[::-1][i-2:])[::-1]

            return out + ' '
        
        return t.whitespace_
    
    # lemmatization with filtering
    ret = ''.join([spacy_token_filter(token) for token in doc])
    return ret.strip().lower()