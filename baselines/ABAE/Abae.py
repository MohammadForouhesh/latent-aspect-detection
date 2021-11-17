import json
import pickle

import pytrec_eval
import pandas as pd


def trec_pickle(path, count):
    with open(path, 'r') as fp:     data = json.load(fp)
    acc = []
    for implicitness_key in data:
        implicitness_level = data[implicitness_key][count]
        print(implicitness_level)
        evaluator = pytrec_eval.RelevanceEvaluator(implicitness_level['qrel'], {'ndcg', 'recall_3', 'map',
                                                                                'success_1', 'success_3',
                                                                                'success_5', 'success_10',
                                                                                'success_32'})

        acc.append(pd.DataFrame(evaluator.evaluate(implicitness_level['qrun'])).mean(axis=1))
    return pd.concat(acc, axis=1)


if __name__ == '__main__':
    ax = []
    sam_eval = ['sam_eval2014.json', 'sam_eval2015.json', 'sam_eval2016.dict.dict']
    count = 2014
    for path in sam_eval:
        trec_pickle(path, str(count)).to_excel(path[:-5]+'.xlsx')
        count += 1
        raise ""