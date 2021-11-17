from source.AspectOpinionOccurrence import lda_occurrence
from source import LDATopicModeling
import pandas as pd
import numpy as np
from source.LDAInference import lda_inference
from sklearn.preprocessing import MinMaxScaler
from source.eval.Evaluation import pytrec_evaluation, report_evaluation, evaluation_experimentation
from source.eval.LatentAspectEvaluation import pxp_eval_indexer

scaler = MinMaxScaler()
converter = lambda array: np.array([item.occurrence for item in array])
convert_matrix = lambda matrix: np.array([converter(array) for array in matrix])

pooling = lambda joint_distribution, pool_size: [np.argsort(joint_distribution[:, j])[::-1][:pool_size]
                                                 for j in range(0, len(joint_distribution[0]))]


def opinionated_pooling_layer(sentence: str, aspect_model: LDATopicModeling.TopicModeling,
                              opinion_model: LDATopicModeling.TopicModeling, oa_pool: list, **kwargs):
    opinion_distro = lda_inference(opinion_model, sentence)
    opinion = np.argmax(opinion_distro)
    oa_pair = oa_pool[opinion]
    aspect_distro = lda_inference(aspect_model, sentence)
    const = aspect_model.lda_model.num_topics
    opinionated_aspect_distro = [aspect_distro[ind] + opinion_distro[opinion]
                                 if ind in oa_pair else aspect_distro[ind] for ind in range(0, const)]

    return np.argsort(opinionated_aspect_distro)[::-1]


def opinionated_aspect_detection(sentence: str, aspect_model: LDATopicModeling.TopicModeling,
                                 opinion_model: LDATopicModeling.TopicModeling, joint_probability: list, theta=0.05, **kwargs):

    opinion_distro = np.array(lda_inference(opinion_model, sentence))
    aspect_distro = np.array(lda_inference(aspect_model, sentence))
    opinion_top_ind = np.argsort(opinion_distro)[:-4:-1]
    opinion_top_vec = opinion_distro[opinion_top_ind]
    olad_joint_part = joint_probability[:, opinion_top_ind]
    probability_kernel = np.inner(olad_joint_part, opinion_top_vec)
    opinionated_aspect_distro = aspect_distro + theta * probability_kernel
    return np.argsort(opinionated_aspect_distro)[::-1]


def opinionated_aspect_extraction(evaluation_set: pd.DataFrame, opinionated_layer_functional, **kwargs) -> pd.DataFrame:
    
    evaluation_set['detected_aspect'] = evaluation_set['None_preprocessed'].apply(opinionated_layer_functional, **kwargs)
    
    return pytrec_evaluation(evaluation_set['sam_eval_aspect'].apply(pxp_eval_indexer, corpus_ic=kwargs['corpus_ic'],
                                                                     model=kwargs['aspect_model']),
                             evaluation_set['detected_aspect'], success_at=kwargs['aspect_model'].lda_model.num_topics)\
            .to_frame()\
            .transpose()


def report_opinionated(sam_eval_test_doc, aspect_model: LDATopicModeling.TopicModeling, opinion_model: LDATopicModeling.TopicModeling,
                       opinionated_layer_functional, corpus_ic: dict, train_set: pd.DataFrame, theta: float = 0.005):

    joint_probability = lda_occurrence(aspect_model, opinion_model, parent_document=train_set)
    oa_pool = pooling(joint_probability, pool_size=1)
    joint_probability_normal = scaler.fit_transform(convert_matrix(joint_probability))
    x, acc = evaluation_experimentation(sam_eval_test_doc, evaluation_functional=opinionated_aspect_extraction,
                                        opinionated_layer_functional=opinionated_layer_functional,
                                        aspect_model=aspect_model,
                                        opinion_model=opinion_model,
                                        corpus_ic=corpus_ic, oa_pool=oa_pool,
                                        joint_probability=joint_probability_normal, theta=theta)
    return report_evaluation(ir_metrics_array=acc)
