import numpy as np
import pandas as pd

from source import LDATopicModeling
from source.eval.Evaluation import pytrec_evaluation
from source.LDAInference import lda_inference, topic_detection


def loc_lda_eval_indexer(aspects, model):
    for aspect in aspects:
        eval_distro = lda_inference(model, aspect)

        yield np.array(eval_distro).argmax() if max(eval_distro) > 1.3 / model.lda_model.num_topics else -1


def loc_lda_evaluation_functional(model: LDATopicModeling.TopicModeling, evaluation_set: pd.DataFrame) -> pd.DataFrame:

    evaluation_set['detected_aspect'] = evaluation_set['None_preprocessed'].apply(topic_detection, model=model)
    return pytrec_evaluation(evaluation_set['sam_eval_aspect'].apply(loc_lda_eval_indexer, model=model),
                             evaluation_set['detected_aspect'], success_at=model.lda_model.num_topics)\
            .to_frame()\
            .transpose()