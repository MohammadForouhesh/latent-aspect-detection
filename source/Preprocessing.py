"""
Create an empty dataset D_prep, with columns caption, aspect_preprocessed, and opinion_preprocessed.

Given dataset D, for every review r in D, save review r in the corresponding row in dataset D_prep under the column
caption, create r_noun from review r, by the following procedure:
 First, remove numerical and non_english data by applying procedures 1 and 2. Then using 3, extract noun and noun phrases and lemmatize them according to their part-of-speech. Doing this part earlier helps lemmatize more accurately. Finally, remove stop-words, emojis, and punctuations by applying 4, 5, and 6.
 Save r_noun in the corresponding row in the dataset D_prep under the column aspect_preprocessed.

Also, create r_adj from review r by the following procedure:
 First, remove numerical and non_english data by applying procedures 1 and 2. Then using 3, extract adjectives and adverbs and lemmatize them according to their part-of-speech. Doing this part earlier helps lemmatize more accurately. Finally, remove stop-words, emojis, and punctuations by applying 4, 5, and 6.
 Save r_noun in the corresponding row in the dataset D_prep under the column opinion_preprocessed.

"""

import re
import gc
import os
from datetime import datetime


import nltk
import spacy
import torch

import string
import pandas as pd
from numpy import percentile
from textblob import TextBlob
from flair.data import Sentence

from nltk import wordnet
from nltk import PorterStemmer
from nltk import WordNetLemmatizer

from thinc.api import set_gpu_allocator, require_gpu
from source.Logging import process_logger

gc.enable()

torch.backends.cudnn.benchmark = True

set_gpu_allocator("pytorch")
require_gpu(0)
torch.cuda.set_per_process_memory_fraction(1.0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cuda:0")


global nlp, sw_spacy, en_words, handy

en_words = set(nltk.corpus.words.words())
handy = {"nt", "do", "na", "n’t", "gf", "JW", "jw", "don't", "didn't", "there", "and", "we've", "’", "t", "n", 'come',
         'try', 'go', 'get', 'make', 'would', 'really', 'like', 'great', 'came', 'got', 'google', "translate",
         'original','lot', 'fun', 'bit', 'hard', 'nan', 'day', 'high', 'big', 'long', 'hot', 'sad', 'easy',
         'hard', 'there', 'away', 've', 'm', 's', "'s", 're'}

try:    nlp = spacy.load("en_core_web_trf")
except:
    os.system('python -m spacy download en_core_web_trf')
    nlp = spacy.load("en_core_web_trf")

sw_spacy = nlp.Defaults.stop_words.union(handy)

if 'serious' in sw_spacy:   sw_spacy.remove('serious')

wnl = WordNetLemmatizer()
ps = PorterStemmer()


# ======================================================================================================================
def dic_lookup(word:str, suggestion:str, pos:str):
    if suggestion in en_words: return suggestion
    try:    return min([w for w in en_words if word in w and nlp(w)[0].pos_ == pos], key=len)
    except: return ""


def synset_similar(word:str, pos:str):
    simil_net = wordnet.wordnet.synsets(word, pos=pos)
    if len(simil_net) == 0: return word
    syn = simil_net[0]
    return syn.name()[:syn.name().find('.')]


def lemma(token):
    if token.pos_ == 'NOUN':
        return token.lemma_
    elif token.pos_ == 'ADJ':
        return wnl.lemmatize(token.text, pos='a')
    elif token.pos_ == 'ADV':
        if len(token.text) > 6 and token.text[-2:] == 'ly':
            word = token.text[:-3]
            suggestion = token.text[:-2]
            return dic_lookup(word, suggestion=suggestion, pos='ADJ')
        return wnl.lemmatize(token.text, pos='r')
    else:
        return token.lemma_


# ======================================================================================================================
def remove_translate(text:str):
    """
    Find the phrase (original) in the text, and remove whatever that comes next to it, and remove
    (original) itself, too.

    :param text:            a string
    :return:                remove the non english part from the input string
    """

    ind = text.lower().find("(original)")
    if ind != -1:   return text[:ind]
    return text


def spell_check(text:str):
    textBlb = TextBlob(text)
    spell_checked = textBlb.correct()
    return str(spell_checked)


def remove_stop_word(text:str, specifications:set=None):
    """
    Given text T, for every word w in T, if w is in the stop-words list of library l1 (spacy), remove w
    from T.

    :param text:            a string, to be stop words removed.
    :param specifications:   a set of words to be added into the list of stop words
    :return:                a string without stop words.
    """

    global sw_spacy
    if specifications is not None:
        sw_spacy = sw_spacy.union(specifications)
    words = [word.lower() for word in text.split() if word.lower() not in sw_spacy and len(word) > 2]
    return " ".join(words)


def remove_emoji(text:str):
    """
    Given text T, for every character char in T, if Unicode of char represents an emoji, remove char from T 
    using library l2 (re).

    :param text:            a string, to be its emojis removed
    :return:                a string without emojis and other irrelebent patterns.
    """

    regex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F300-\U0001F5FF"
                                        u"\U0001F1E6-\U0001F1FF"
                                        u"\U00002700-\U000027BF"
                                        u"\U0001F900-\U0001F9FF"
                                        u"\U0001F600-\U0001F64F"
                                        u"\U0001F680-\U0001F6FF"
                                        u"\U00002600-\U000026FF"
                                        "]+", flags=re.UNICODE)
    return str(regex_pattern.sub(r'', text))


def flair_lemmatize(text:str, tagger, allowed_postags='aspect'):
    """
    Lemmatization and Stemming using flair model:
        Make use of part of speech tagging in lemmatization as follows
            Given text T, for every word w in T, find POS of w using flair. Now consider
            the following cases:
            1     if w is a noun, then use library l4 (spacy) for lemmatization.
            2     if w is an adjective, use library l5 (nltk wordnet lemmatizer) with keyword 
                  argument pos equal to 'a'.
            3     if w is an adverb, the consider the following cases:
                    * if w contains more than six characters and ends with 'ly', then removing 
                      the last three characters of w gives w_head, now find the shortest word 
                      in the English dictionary of library l6 (nltk corpus words) that contains 
                      w_head.
                    * else use l5 (nltk wordnet lemmatizer) with keyword argument pos equal to 'r'.
            4     if none of the above cases were applicable, then use library l4 (spacy) for lemmatization.

    :param text:            a string to be lemmatized
    :param tagger:          tagger, pretrained flair model instance.
    :param allowed_postags: which postags should remain, choose from the following options:
                            1       'aspect' if you want to keep noun phrases
                            2       'opinion' if you want to keep adjectives and verbs
                            3       'all' if you want both aspect and opinions
                            4       None if you just want to lemmatize the input string

    :return:                an output lemmatized string
    """

    flair_doc = Sentence(text)
    tagger.predict(flair_doc)
    spacy_doc = nlp(flair_doc.to_tokenized_string())
    if allowed_postags   == 'aspect':  allowed_postags = ['NN', 'NNP', 'NNS']                       # for aspect
    elif allowed_postags == 'opinion': allowed_postags = ['JJ', 'JJR', 'JJS', #'RB', 'RBR', 'RBS',
                                                          'VB', 'VBG', 'VBN', 'VBP', 'VBZ']         # for opinion
    elif allowed_postags == 'all': allowed_postags = ['NN', 'NNP', 'NNS', 'JJ', 'JJR', 'JJS',
                                                      'VB', 'VBG', 'VBN', 'VBP', 'VBZ']

    else: return ' '.join([token.lemma_ for token in spacy_doc])
    flair_lemmatized = [lemma(spacy_doc[entity.idx - 1]) for entity in flair_doc
                    if entity.get_tag('pos').value in allowed_postags]
    return " ".join(flair_lemmatized)


def spacy_lemmatize(text:str, allowed_postags='aspect'):
    """
    Lemmatization and Stemming using spacy:
        Make use of part of speech tagging in lemmatization as follows
            Given text T, for every word w in T, find POS of w using flair. Now consider
            the following cases:
            1     if w is a noun, then use library l4 (spacy) for lemmatization.
            2     if w is an adjective, use library l5 (nltk wordnet lemmatizer) with keyword 
                  argument pos equal to 'a'.
            3     if w is an adverb, the consider the following cases:
                    * if w contains more than six characters and ends with 'ly', then removing 
                      the last three characters of w gives w_head, now find the shortest word 
                      in the English dictionary of library l6 (nltk corpus words) that contains 
                      w_head.
                    * else use l5 (nltk wordnet lemmatizer) with keyword argument pos equal to 'r'.
            4     if none of the above cases were applicable, then use library l4 (spacy) for lemmatization.

    :param text:            a string to be lemmatized
    :param allowed_postags: which postags should remain, choose from the following options:
                            1       'aspect' if you want to keep noun phrases
                            2       'opinion' if you want to keep adjectives and verbs
                            3       'all' if you want both aspect and opinions
                            4       None if you just want to lemmatize the input string

    :return:                an output lemmatized string
    """
    text = str(text)
    doc = nlp(text)

    if allowed_postags   == 'aspect':   allowed_postags = ['NOUN', 'PROPN']               # for aspect
    elif allowed_postags == 'opinion':  allowed_postags = ['ADJ', 'VERB']                 # for opinion
    elif allowed_postags == 'all':      allowed_postags = ['NOUN', 'PROPN', 'ADJ', 'VERB']
    else: return ' '.join([token.lemma_ for token in doc])

    lemmatized = [lemma(token) for token in doc if token.pos_ in allowed_postags]
    try:    lemmatized_text = " ".join(lemmatized)
    except: lemmatized_text = ""

    return lemmatized_text


def remove_punctuation(text:str):
    """
    Given text T, for every character char in T, if char is a punctuation mark, remove char from T using 
    built-in tools of python strings.

    :param text:            an input string, with punctuation
    :return:                text wordout punctuation
    """

    text = text.replace('.', '. ').replace('!', '! ').replace(',', ', ').replace('?', '? ').replace(':', ': ')
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in text.split()]
    return " ".join(words)


def anglicize(text:str):
    anglicized = [w for w in nltk.wordpunct_tokenize(text) if w.lower() in en_words or not w.isalpha()]
    return " ".join(anglicized)


def numerics(text:str):
    """
    Takes a string as input, removes digits and numerical data in the text, and returns it.
    :param text:            input string
    :return:                output string without numerical data.
    """
    return ''.join([char for char in text if not char.isdigit()])


def preprocess(text:str, tagger=None, postag:str='aspect'):
    text = remove_translate(text)
    if len(text) > 0:
        if tagger is None:  text = spacy_lemmatize(text, allowed_postags=postag)
        else:               text = flair_lemmatize(text, tagger=tagger, allowed_postags=postag)
    if len(text) > 0:       text = remove_stop_word(text)
    if len(text) > 0:       text = remove_emoji(text)
    text = numerics(str(text))
    return remove_punctuation(text)


# ======================================================================================================================
def segmentation(dataframe:pd.DataFrame):
    array = []
    series = dataframe['caption']
    for i in range(0, dataframe.shape[0]):
        item = nlp(series[i].replace("!", ". ").replace(".", ". ").replace("?", ". "))
        assert item.has_annotation("SENT_START")
        array += [[dataframe['username'].iloc[i], dataframe['absolute_date'].iloc[i], sent.text]\
                  for sent in item.sents if len(sent.text) > 2]

    return pd.DataFrame(array, columns=dataframe.columns)


def augmentation(dataframe:pd.DataFrame, augment:int=10):
    return pd.concat([dataframe]*augment, ignore_index=True)


def specific_stop_words(dataframe:pd.DataFrame, column:str) -> set:
    """
    After calculating opinion_preprocessed and aspect_preprocessed columns, proceed with
    content-aware stop-words removal for each column:

    Content-Aware Stop-words:

        Given column D_preprocessed, calculate the count of each word in the whole document. 
        For an arbitrary word w, if the count of w is in the 99.9 percentile, then add this 
        word to a set sw_specific. For every word w in D_preprocessed, if w in sw_specific,
        then remove w from D_preprocessed.

    :param dataframe:       pandas dataframe
    :param column:          column name, the input dataframe must contain this column.
    :return:                a set of expendable words, to be treated as stop words
    """
    counts = dataframe[column].str.split(expand=True).stack().value_counts()
    specification = set(counts[:len([count for count in counts if count > percentile(counts, 99.9)])].index)
    return specification


def preprocess_in_place(data_frame: pd.DataFrame, column_name: str, postags: list):
    time_stamp = datetime.now()
    for postag in postags:
        process_logger(time_stamp, '', postags.index(postag)/len(postags))
        data_frame['{}_preprocessed'.format(postag)] = data_frame[column_name].apply(preprocess, tagger=None, postag=postag)
    return data_frame