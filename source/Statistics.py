from datetime import datetime

import numpy as np
import pandas as pd
from nltk.corpus import wordnet_ic

from source import LDATopicModeling
from source.SamEval import read_sam_eval
from source.LDAInference import topic_detection, lda_mount, word_topic_detection, lda_inference
from source.Logging import process_logger
from source.Preprocessing import preprocess


def sam_eval_aspect_distribution(model: LDATopicModeling.TopicModeling, sam_eval_doc, corpus_ic):
    prep = sam_eval_doc['caption'].apply(preprocess, postag=None, tagger=None)
    topic_detection_series = prep.apply(topic_detection, model=model).apply(lambda item: item[0])
    for ind, aspects in enumerate(prep):
        process_logger(datetime.now(), 'sam_evaling', ind/len(prep))
        eval_index = []
        for aspect in aspects:
            word_topic = word_topic_detection(word=aspect, model=model, corpus_ic=corpus_ic)
            eval_distro = lda_inference(model, aspect)

            eval_index.append(np.array(eval_distro).argmax() if max(eval_distro) > 1.3 / model.lda_model.num_topics
                              else word_topic[0] if word_topic[1] > 2 else -1)

    return pd.DataFrame([topic_detection_series.value_counts().sort_index(), pd.Series(eval_index).value_counts().sort_index], index=['que', 'pxp']).transpose()


def unique_words_counter(dataset):
    return pd.DataFrame([word for line in dataset for word in line]).count()


sam_eval_aspect_distribution(model=lda_mount('../models/pxp_model_flair_aspect_Canadian_Casinos.pxp'),
                             sam_eval_doc=read_sam_eval('../sam_eval/sam_eval2014.txt'),
                             corpus_ic=wordnet_ic.ic('ic-brown.dat')).to_excel('../aspect_distro_report.xlsx')

print(unique_words_counter(dataset=read_sam_eval('../sam_eval/sam_eval2014.txt')['caption'])\
                                            .apply(preprocess, postag='aspect', tagger=None))

print(unique_words_counter(dataset=read_sam_eval('../sam_eval/sam_eval2014.txt')['caption'])\
                                            .apply(preprocess, postag='opinion', tagger=None))


