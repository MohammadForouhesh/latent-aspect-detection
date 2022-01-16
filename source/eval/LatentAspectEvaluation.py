import numpy as np
import pandas as pd

from source import LDATopicModeling
from source.LDAInference import word_topic_detection, lda_inference, topic_detection
from source.eval.Evaluation import pytrec_evaluation


def pxp_eval_indexer(aspects, corpus_ic, model):
    for aspect in aspects:
        word_topic = word_topic_detection(word=aspect, model=model, corpus_ic=corpus_ic)
        eval_distro = lda_inference(model, aspect)

        yield np.array(eval_distro).argmax() if max(eval_distro) > 1.3 / model.lda_model.num_topics \
                                             else word_topic[0] if word_topic[1] > 2 else -1


def hidden_aspect_evaluation(model: LDATopicModeling.TopicModeling, evaluation_set: pd.DataFrame, corpus_ic: dict) -> pd.DataFrame:

    evaluation_set['detected_aspect'] = evaluation_set['None_preprocessed'].apply(topic_detection, model=model)
    return pytrec_evaluation(evaluation_set['sem_eval_aspect'].apply(pxp_eval_indexer, corpus_ic=corpus_ic, model=model),
                             evaluation_set['detected_aspect'], success_at=model.lda_model.num_topics)\
            .to_frame()\
            .transpose()