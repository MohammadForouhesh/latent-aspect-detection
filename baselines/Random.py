import numpy as np
import pandas as pd

from source.eval.Evaluation import pytrec_evaluation


def random_word_choosing(sentence: str):
    return [np.random.randint(len(sentence.split(' '))) for _ in range(0, 32)]


def random_evaluation_functional(evaluation_set: pd.DataFrame):
    evaluation_set['detected_aspect'] = evaluation_set['caption'].apply(random_word_choosing)
    return pytrec_evaluation(evaluation_set['aspect_index'].apply(lambda item: item[-1]), evaluation_set['detected_aspect'], success_at=100) \
        .to_frame() \
        .transpose()
