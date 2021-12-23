import pandas as pd
import numpy as np
import pytrec_eval
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from source.Logging import process_logger
from source.Preprocessing import preprocess


def aspect_remover(series: pd.Series):
    splited = series.caption.split(" ")
    rand_ind = np.random.randint(len(series.aspect_index))
    trimmed = [word for word in splited if splited.index(word) not in series.aspect_index[rand_ind]]
    return preprocess(" ".join(trimmed), tagger=None, postag=None)


def pytrec_evaluation(sam_eval_index_aspect:pd.Series, detected_aspect:pd.Series, success_at) -> pd.Series:

    pytrec_qrel = {}
    pytrec_qrun = {}
    for ind, eval_index in enumerate(sam_eval_index_aspect):

        pytrec_qrel['query'+str(ind)] = {'topic'+str(eval_j):success_at-j for j, eval_j in enumerate(eval_index)}
        pytrec_qrun['query'+str(ind)] = {'topic'+str(detected_aspect.iloc[ind][i]):success_at-i
                                         for i in range(0, len(detected_aspect.iloc[ind]))}

    evaluator = pytrec_eval.RelevanceEvaluator(pytrec_qrel, {'ndcg', 'recall_5', 'recip_rank', 'success_1', 'success_3', 'success_5',
                                                             'success_10', 'success_32'})

    return pd.DataFrame(evaluator.evaluate(pytrec_qrun)).mean(axis=1)


def evaluation_setup(dataset: pd.DataFrame, percentage: float = 0.1, random_state=42) -> pd.DataFrame:
    if percentage == 0:     evaluation_set = dataset
    else:
        if percentage == 1: hidden_set = dataset; visible_set = pd.DataFrame()
        else:  visible_set, hidden_set = train_test_split(dataset, test_size=percentage, random_state=random_state)
    
        hidden_set['None_preprocessed'] = hidden_set.apply(aspect_remover, axis=1)
        evaluation_set = visible_set.append(hidden_set, ignore_index=True)
        
    return evaluation_set


def evaluation_experimentation(baseline_dataset, evaluation_functional, **kwargs):
    kf = KFold(5, shuffle=True, random_state=42)
    acc = []
    grid_size = [round(0.05 * x, 2) for x in range(0, 20)]
    for ind in range(0, len(grid_size)):
        acc_fold = []
        process_logger('evaluation', ratio=ind/len(grid_size))
        for _, train_ind in kf.split(baseline_dataset):
            X_cv = baseline_dataset.iloc[train_ind]
            evaluation_set = evaluation_setup(dataset=X_cv, percentage=grid_size[ind])
            acc_fold.append(evaluation_functional(evaluation_set=evaluation_set, **kwargs))
        acc.append(pd.concat(acc_fold, ignore_index=True))
        
    return grid_size, acc


def report_evaluation(ir_metrics_array):
    for ind, metric in enumerate(['ndcg', 'recall_5', 'recip_rank', 'success_1', 'success_3', 'success_5', 'success_10', 'success_32']):
        mean = [ir_metrics_array[i][metric].mean() for i in range(0, len(ir_metrics_array))]
        std = [ir_metrics_array[i][metric].std() for i in range(0, len(ir_metrics_array))]
        yield ['{:.2f}% ± {:.2f}'.format(mean[ind]*100, std[ind]*100) for ind in range(0, len(mean))]
        # yield [round(mean[ind] * 100, 2) for ind in range(0, len(mean))]


def report_pure(sam_eval_test_doc, evaluation_functional, **kwargs):
    x, acc = evaluation_experimentation(sam_eval_test_doc, evaluation_functional, **kwargs)
    return report_evaluation(ir_metrics_array=acc)
