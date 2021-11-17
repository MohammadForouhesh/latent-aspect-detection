# -*- coding: utf-8 -*-

# ! Users\Mohammad.FT\PycharmProjects\PxP-TopicModeling python -W ignore::DeprecationWarning

"""
Created on Thur Mar 11 21:11:32 2021

@author: Mohammad.FT
"""
import gc
import os
import nltk
from nltk.corpus import wordnet_ic
from baselines.KMeans import AspectKMeans, akmeans_evaluation_functional
from baselines.LocLDA import loc_lda_evaluation_functional
from baselines.Random import random_evaluation_functional
from source.SamEval import read_sam_eval
from source.eval.Evaluation import report_pure
from source.eval.LatentAspectEvaluation import hidden_aspect_evaluation
from source.eval.OpinionatedEvaluation import report_opinionated, opinionated_pooling_layer, \
    opinionated_aspect_detection

nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('wordnet_ic', quiet=True)
import gensim
import pickle5 as pickle
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from termcolor import colored
from source.Logging import logger
from source import LDATopicModeling
from flair.models import SequenceTagger
from source.OpinionSpecification import correction
from source.LDATopicModeling import TopicModeling, elbow_method
from source.Preprocessing import preprocess, remove_stop_word, specific_stop_words, preprocess_in_place

gc.enable()
warnings.filterwarnings("ignore", category=DeprecationWarning)


def corpus_preparation(dictionary, series: pd.Series) -> list:
    string = [str(line).split() for line in series if line is not np.nan]
    corpus = [dictionary.doc2bow(text) for text in string]
    return corpus


def builder(args, train_series: pd.Series, postag: str) -> LDATopicModeling.TopicModeling:
    pxp_model = TopicModeling(train_series, bigram=True)  # bool(postag=='opinion'))
    print(colored("model " + args.engine + " is built.", 'cyan'))
    model_name = postag + '_' + args.path[args.path.find('/') + 1:args.path.find('.xlsx')]
    # ==================================================================================================================
    if args.tune:
        coherence_values = pxp_model.cross_validation(start=5, limit=100, step=3)
        print(coherence_values)
        optimal_topic = elbow_method(coherence_values, name=model_name, start=5, limit=100, step=3)
        print(colored("Our novel tuning system detected #" + str(optimal_topic) + " with coherence " + \
                      " as the optimal value for topic number", 'cyan'))

        args.num_topics = optimal_topic

    # ==================================================================================================================
    pxp_model.topic_modeling(num_topics=args.num_topics, library=args.engine, alpha=args.alpha,
                             iterations=args.iterations)
    string = 'models/pxp_model_flair_' + model_name + '.pxp'
    with open(string, 'wb') as handle:
        pickle.dump(pxp_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pxp_model


def prep_sam_eval() -> dict:
    sam_eval_paths = {'201{}'.format(i): 'sam_eval/sam_eval201{}.txt'.format(i) for i in [4, 5, 6]}
    sam_eval_frame = {key: preprocess_in_place(read_sam_eval(sam_eval_paths[key]),
                                                               column_name='caption',
                                                               postags=['aspect', 'opinion', 'all', None])
                      for key in sam_eval_paths}
    with open('sam_eval/preprocessed_data_frame.dict', 'wb') as handle:
        pickle.dump(sam_eval_frame, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sam_eval_frame


def main(args):
    start_time = datetime.now()
    print(logger(datetime.now(), 'start preprocessing', 'sam_eval dataset ready soon'))

    if os.path.isfile('sam_eval/preprocessed_data_frame.dict'):
        with open('sam_eval/preprocessed_data_frame.dict', 'rb') as handle:
            sam_eval_dict = pickle.load(handle)
    else:   sam_eval_dict = prep_sam_eval()

    print(logger(datetime.now(), 'end preprocessing', ''))
    dataset = pd.read_excel(args.path, index_col=0)

    nan_value = float("NaN")
    # ==================================================================================================================
    prep_n_seg_path = 'prep_and_seg_datasets/' + args.path[args.path.find('/') + 1:args.path.find('.xlsx')].replace('/', '')
    os.makedirs(prep_n_seg_path, exist_ok=True)
    if args.preprocess:
        if args.flair:  flair_tagger = SequenceTagger.load("flair/pos-english-fast")
        else:           flair_tagger = None

        print(logger(datetime.now(), 'start preprocessing', 'pxp dataset takes several hours'))
        for postag in args.postag_list:
            print(logger(datetime.now(), postag, ''))
            if postag == 'all': continue
            dataset[postag + '_preprocessed'] = dataset['caption'].apply(preprocess, tagger=flair_tagger, postag=postag)
            specification = specific_stop_words(dataset, column=postag + '_preprocessed')
            dataset[postag + '_preprocessed'] = dataset[postag + '_preprocessed'] \
                .apply(remove_stop_word, specifications=specification)

            dataset.replace("", nan_value, inplace=True)
            dataset.replace(np.nan, nan_value, inplace=True)
            dataset.dropna(subset=[postag + '_preprocessed'], how='any', axis='index', inplace=True)
        dataset['all_preprocessed'] = dataset['aspect_preprocessed'] + ' ' + dataset['opinion_preprocessed']

        dataset.to_excel(prep_n_seg_path + '/preprocessed.xlsx')

        print(logger(datetime.now(), 'end preprocessing', ''))

    if args.correction:
        dataset['opinion_preprocessed'] = correction(dataset, pretrained='labels/yelp_all_pretrained_dataset.xlsx')
        dataset.to_excel(prep_n_seg_path + '/preprocessed_corrected.xlsx')
    sam_eval_test_dataset = sam_eval_dict.pop(args.sam_eval_test)
    dataset = pd.concat([dataset] + [sam_df for sam_df in sam_eval_dict.values()]).fillna("sam_eval")

    # ==================================================================================================================
    lda_storage = {}
    for postag in args.postag_list:
        model_args = getattr(args, postag + '_model')
        print(logger(datetime.now(),  '{} lda'.format(postag), 'using pre-trained model'\
                                                                if model_args is not None else 'building lda models'))

        if model_args is None:  model = builder(args, dataset[postag + '_preprocessed'], postag=postag)
        else:
            with open(model_args, 'rb') as handle:
                model = pickle.load(handle)
        model.lda_model = gensim.models.wrappers.ldamallet. \
            malletmodel2ldamodel(model.lda_model, gamma_threshold=0.001, iterations=50)
        lda_storage[postag] = model

    # ==================================================================================================================
    print(logger(datetime.now(), 'create analytics table', ''))
    from source.AspectOpinionOccurrence import occurrence_builder, topic_relevance
    if args.labeling is not None:   labeling_doc = pd.read_excel(args.labeling, header=None).transpose().to_numpy(na_value="")
    else:                           labeling_doc = None
    occurrence_builder(lda_storage['aspect'], lda_storage['opinion'], lda_storage['all'], parent_document=dataset,
                       save_path='pipeline/' + args.path[:args.path.find('.xlsx')].replace('/', '_') + '_report_community.xlsx',
                       labeling_doc=labeling_doc)

    topic_relevance(lda_storage['aspect'], lda_storage['opinion'], parent_document=dataset,
                    save_path='pipeline/' + args.path[:args.path.find('.xlsx')].replace('/', '_') + '_report_community_details.xlsx')

    # ==================================================================================================================
    print(logger(datetime.now(), 'evaluate latent aspect', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))

    pd.DataFrame(report_pure(sam_eval_test_dataset, evaluation_functional=hidden_aspect_evaluation,
                             model=lda_storage['aspect'], corpus_ic=wordnet_ic.ic('ic-brown.dat')),
                 index=['ndcg','recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32'])\
                .to_excel('reports/report_pure_{}.xlsx'.format(args.sam_eval_test))

    print(logger(datetime.now(), 'evaluate opinionated aspect', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    pd.DataFrame(report_opinionated(sam_eval_test_dataset, aspect_model=lda_storage['aspect'],
                                    opinion_model=lda_storage['opinion'],
                                    opinionated_layer_functional=opinionated_pooling_layer,
                                    corpus_ic=wordnet_ic.ic('ic-brown.dat'),
                                    train_set=dataset),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32'])\
                .to_excel('reports/report_opinionated_pool_{}.xlsx'.format(args.sam_eval_test))

    print(logger(datetime.now(), 'evaluate opinionated aspect', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    pd.DataFrame(report_opinionated(sam_eval_test_dataset, aspect_model=lda_storage['aspect'],
                                    opinion_model=lda_storage['opinion'],
                                    opinionated_layer_functional=opinionated_aspect_detection,
                                    corpus_ic=wordnet_ic.ic('ic-brown.dat'),
                                    train_set=dataset, theta=0.1),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32'])\
                 .to_excel('reports/report_opinionated_matrix_theta1_{}.xlsx'.format(args.sam_eval_test))


    print(logger(datetime.now(), 'evaluate opinionated aspect', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    pd.DataFrame(report_opinionated(sam_eval_test_dataset, aspect_model=lda_storage['aspect'],
                                    opinion_model=lda_storage['opinion'],
                                    opinionated_layer_functional=opinionated_aspect_detection,
                                    corpus_ic=wordnet_ic.ic('ic-brown.dat'),
                                    train_set=dataset, theta=0.2),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32'])\
                .to_excel('reports/report_opinionated_matrix_theta2_{}.xlsx'.format(args.sam_eval_test))


    print(logger(datetime.now(), 'evaluate opinionated aspect', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    pd.DataFrame(report_opinionated(sam_eval_test_dataset, aspect_model=lda_storage['aspect'],
                                    opinion_model=lda_storage['opinion'],
                                    opinionated_layer_functional=opinionated_aspect_detection,
                                    corpus_ic=wordnet_ic.ic('ic-brown.dat'),
                                    train_set=dataset, theta=0.005),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32'])\
                .to_excel('reports/report_opinionated_matrix_theta005_{}.xlsx'.format(args.sam_eval_test))
    print(logger(datetime.now(), 'evaluate baseline Random', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    pd.DataFrame(report_pure(sam_eval_test_dataset, evaluation_functional=random_evaluation_functional),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32'])\
        .to_excel('reports/report_random_{}.xlsx'.format(args.sam_eval_test))

    print(logger(datetime.now(), 'train/eval baseline KMeans', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    kmeans_model = AspectKMeans(8)
    kmeans_model.train(dataset['caption'])
    pd.DataFrame(report_pure(sam_eval_test_dataset, evaluation_functional=akmeans_evaluation_functional, model=kmeans_model),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32']) \
        .to_excel('reports/report_kmeans_{}.xlsx'.format(args.sam_eval_test))

    print(logger(datetime.now(), 'train/eval baseline LocLDA', 'testing sam_eval{} restaurant dataset'.format(args.sam_eval_test)))
    locLDA = TopicModeling(dataset.all_preprocessed, bigram=False)
    locLDA.topic_modeling(num_topics=32, library='mallet', iterations=args.iterations)
    locLDA.lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(locLDA.lda_model,
                                                                             gamma_threshold=0.001,
                                                                             iterations=50)
    pd.DataFrame(report_pure(sam_eval_test_dataset, evaluation_functional=loc_lda_evaluation_functional, model=locLDA),
                 index=['ndcg', 'recall_3', 'map', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32']) \
        .to_excel('reports/report_locLDA_{}.xlsx'.format(args.sam_eval_test))
    print(colored(datetime.now() - start_time, 'cyan'))


if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("picture", exist_ok=True)
    os.makedirs("logging", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("pipeline", exist_ok=True)
    os.makedirs("inference", exist_ok=True)
    os.makedirs("prep_and_seg_datasets", exist_ok=True)
    # ==================================================================================================================
    parser = argparse.ArgumentParser(description='PXP Topic Modeling.')
    parser.add_argument('--path', dest='path', type=str, default='data/Canadian_Casinos.xlsx',
                        help='Raw dataset file address.')
    parser.add_argument('--segment', dest='segment', type=bool, default=True, help='break every record into sentences.')
    parser.add_argument('--augment', dest='augment', type=int, default=None,
                        help='augment the dataset to learn better.')
    parser.add_argument('--engine', dest='engine', type=str, default='mallet',
                        help="supported topic modeling engines in this implementation are mallet, gensim, HDP.")
    parser.add_argument('--preprocess', dest='preprocess', type=bool, default=True,
                        help="whether or not preprocessing documents to be used instead of preprocessing"
                             "the raw document in the --path.")
    parser.add_argument('--tune', dest='tune', type=bool, default=False,
                        help="Using 5 folds of data to detect the best number of topics.")
    parser.add_argument('--num_topics', dest='num_topics', type=int, default=32, help='User defined number of topics.')
    parser.add_argument('--aspect_model', dest='aspect_model', type=str, default=None,
                        help="address to pxp lda model trained on noun documents.")
    parser.add_argument('--opinion_model', dest='opinion_model', type=str, default=None,
                        help="address to pxp lda model trained on adj documents.")
    parser.add_argument('--all_model', dest='all_model', type=str, default=None,
                        help="address to pxp lda model trained on all the phrases of the documents.")
    parser.add_argument('--inference', dest='inference', type=str, default=None,
                        help="address to inference dataset.")
    parser.add_argument('--correction', dest='correction', type=bool, default=True,
                        help="Using yelp dataset to improve bigram detection.")
    parser.add_argument('--flair', dest='flair', type=bool, default=True,
                        help="Using flair POS Tagger in the preprocessing unit.")
    parser.add_argument('--postag_list', dest='postag_list', type=list, default=["aspect", "opinion", "all"],
                        help="Give a list of possible extraction strategies, from 'aspect', 'opinion', 'all'")
    parser.add_argument('--iterations', dest='iterations', type=int, default=1000,
                        help="mallet parameters")
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help="mallet parameters")
    parser.add_argument('--labeling', dest='labeling', type=str, default=None,
                        help="Customizable labeling of the aspect opinion table, provide tha path to your labeling doc")
    parser.add_argument('--sam_eval_test', dest='sam_eval_test', type=str, default='2014',
                        help="choose between 2014, 2015, 2016. to be tested by our model")

    parser.set_defaults(augment=None, segment=True, tune=False, preprocess=False, flair=False, correction=False,
                        path='data/Canadian_Casinos_preprocessed_corrected.xlsx',
                        #postag_list=["aspect", "opinion", 'all'], #iterations=400, alpha=0.1,
                        #labeling='labels/arash_labels.xlsx',
                        aspect_model='models/pxp_model_flair_aspect_Canadian_Casinos_preprocessed_corrected.pxp',
                        opinion_model='models/pxp_model_flair_opinion_Canadian_Casinos_preprocessed_corrected.pxp',
                        all_model='models/pxp_model_flair_all_Canadian_Casinos_preprocessed_corrected.pxp')

    arguments = parser.parse_args()
    with warnings.catch_warnings():
        logging.basicConfig(filename='logging/' + arguments.path[:arguments.path.find('.xlsx')].replace('/', '_') + \
                                      str(arguments.sam_eval_test) + '_pxp_info.log', format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        warnings.filterwarnings("ignore")
        print(logger(datetime.now(), "", 'PXP=TopicModeling'))
        main(args=arguments)
