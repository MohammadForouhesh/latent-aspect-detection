import os.path

from source.LDATopicModeling import TopicModeling
from source.Preprocessing import preprocess
import pandas as pd
import numpy as np
import gensim


replace_element_list = lambda sentence, word: " ".join([item if item not in word else word for item in sentence.split(" ")])
bag_creator = lambda series: [line.lower().split() for line in series if line is not np.nan]


def make_bigrams(bag_of_word:list, bi_min:int=10, threshold:int=3):
    assert isinstance(bag_of_word, (list, np.ndarray))
    
    bigram = gensim.models.Phrases(bag_of_word, min_count=bi_min, threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def information_extraction(bag_of_word:list, bigram_model:gensim.models.phrases.Phraser) -> list:
    assert isinstance(bag_of_word, list) and isinstance(bigram_model, gensim.models.phrases.Phraser)
    return [bigram_model[review] for review in bag_of_word]
    

def interchange(row:pd.Series) -> pd.Series:
    assert isinstance(row, pd.Series)
    
    for word in row.interchange_list:
        row.opinion_preprocessed = replace_element_list(row.opinion_preprocessed, word)
    return row
    
    
def correction(dataframe:pd.DataFrame, pretrained:str=None) -> pd.Series:
    assert isinstance(dataframe, pd.DataFrame)
    assert os.path.isfile(pretrained), 'No such file exists'
    
    prep_caption_bag = bag_creator(dataframe.caption.apply(preprocess, postag='all'))
    if pretrained is not None:  train_bag = bag_creator(pd.read_excel(pretrained).text)
    else:                       train_bag = prep_caption_bag
    
    bigram_model = make_bigrams(bag_of_word=train_bag)
    
    dataframe['interchange_list'] = information_extraction(bag_of_word=prep_caption_bag, bigram_model=bigram_model)
    
    return dataframe.apply(interchange, axis=1).opinion_preprocessed

