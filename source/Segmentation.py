from source.LDAInference import lda_mount, lda_inference
from source.Preprocessing import preprocess, spacy_lemmatize
from source.Logging import print_process_logger
from datetime import datetime
from collections.abc import Generator
from source import LDATopicModeling
from scipy.stats import entropy
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from scipy import linalg as la


pivot_augmentation = lambda doc, frame, col, ind: [[frame[label].iloc[ind] for label in col] + [text] for text in doc
                                                   if len(text) > 3]

cross_product = lambda vector1, vector2: pd.DataFrame(np.array(vector1).reshape(-1, 1) * np.array(vector2))
scattering = lambda model_list, segment: cross_product(lda_inference(model_list[0], segment),
                                                       lda_inference(model_list[1], segment))

matrix_log = lambda matrix: la.logm(matrix)/la.logm(np.asarray([[2]]))
sign = lambda element: 1 if element >= 0 else 0 if element is float('NaN') else -1
bounded_above = lambda element: element if abs(element) < float('inf') else sign(element) * 1e2
bounded_below = lambda element: element if element is not float('NaN') else 0
bounded = lambda element: bounded_above(bounded_below(element))

string_indexing = lambda text: [(key, word) for key, word in enumerate(spacy_lemmatize(text, allowed_postags=None).split(' '))]


def alignment(indexed_text: list, excerpt: str):
    excerpt = excerpt.split(' ')
    for key, word in indexed_text:
        if word in excerpt:
            yield key
            excerpt.pop(excerpt.index(word))


def sentencizer(text, alignment_list):
    begin = 0
    for index in alignment_list:
        yield ' '.join(text.split(' ')[begin:index])
        begin = index


def squarize(matrix: np.ndarray):
    shape = max(matrix.shape)
    nested_list = list(map(list, matrix))
    for row in nested_list:
        yield row + [0] * (shape - len(row))
    for i in range(len(nested_list), shape):
        yield [0] * shape


def quantum_relative_entropy(frame: pd.DataFrame, base: pd.DataFrame):
    bounded_square_frame = list(squarize(frame.applymap(bounded).to_numpy()))
    bounded_square_base = list(squarize(base.applymap(bounded).to_numpy()))
    return np.trace(frame * (matrix_log(bounded_square_frame) - matrix_log(bounded_square_base)))


def shooting_entropy(document: list, **kwargs) -> list:
    """
    :param document: list of strings
    :param kwargs: model_1, model_2 :LDATopicModeling.TopicModeling
    :return: list of float
    """

    if len(document) == 0: return [0]

    try:    model_1 = kwargs['model_1']
    except: raise Exception('At least one model should be passed into the function')

    try:    model_2 = kwargs['model_2']
    except: model_2 = None

    if model_2 is not None:
        base = scattering([model_1, model_2], document[0])
        return [quantum_relative_entropy(scattering([model_1, model_2], document[ind]), base) for ind in range(1, len(document))]
    else:
        base = lda_inference(model_1, document[0])
        return [entropy(lda_inference(model_1, document[ind]), qk=base) for ind in range(1, len(document))]


def word_increment_sentencizer(text: str) -> list:
    word_list = text.split(' ')
    return [preprocess(' '.join(word_list[:ind]), postag='all') for ind in range(0, len(word_list)+1)]


def lda_entropy_segmentation(text: str, alignment_list: list, **kwargs) -> Generator:
    incremental_document = list(sentencizer(text, alignment_list))
    entropy_mask = shooting_entropy(incremental_document, **kwargs)
    const_anomal = np.mean(entropy_mask) + 3 * np.std(entropy_mask)
    index = len([incremental_document[i] for i in range(1, len(incremental_document)) if entropy_mask[i-1] <= const_anomal])
    maimed_text = ' '.join(text.split(' ')[index-1:])
    if len(maimed_text) > 0 and len(alignment_list) > 0:
        yield from lda_entropy_segmentation(maimed_text, alignment_list, **kwargs)
        alignment_list.pop(0)
    yield ' '.join(text.split(' ')[:index-1])


def lda_kmeans_entropy_segmentation(text: str,  **kwargs):
    n_clusters = len(text.split(' '))//8 + 1
    incremental_document = list(word_increment_sentencizer(text))
    entropy_mask = shooting_entropy(incremental_document, **kwargs)
    X = np.array([[i, entropy_mask[i]] for i in range(0, len(entropy_mask))])
    k = KMeans(n_clusters=n_clusters).fit(X)
    mask = k.labels_
    clusters = {label: [] for label in set(mask)}
    for key, item in enumerate(X):
        clusters[mask[key]].append(text.split(' ')[int(item[0])])
    return [' '.join(clusters[key]) for key in clusters]


def review_segmentation(review: str, **kwargs):
    base = preprocess(review, postag='all')
    prep_align = list(alignment(string_indexing(review), base))
    return list(lda_entropy_segmentation(review, prep_align, **kwargs))


def pivot_segmentation(pivot_series: pd.Series, base_dataframe: pd.DataFrame, columns: list, **kwargs):
    time_stamp = datetime.now()
    for index in range(0, base_dataframe.shape[0]):
        print_process_logger(time_stamp, 'segmentation', index/base_dataframe.shape[0])
        prep_align = alignment(pivot_series.iloc[index], base_dataframe['all_preprocessed'].iloc[index])
        document = lda_entropy_segmentation(pivot_series.iloc[index], prep_align, **kwargs)
        yield pivot_augmentation(doc=document, frame=base_dataframe, col=columns, ind=index)


def segmentation(dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
    columns = list(dataframe.columns)
    columns.remove('caption')
    columns.remove('Unnamed: 0') if 'Unnamed: 0' in columns else None
    series = dataframe['caption']
    return pd.DataFrame(pivot_segmentation(series, dataframe, columns, **kwargs), columns=columns)


if __name__ == '__main__':
    all_model = lda_mount('../models/pxp_model_flair_all_Canadian_Casinos.pxp')
    #sentence = "I called today got no answer and the machines were full so hung up on. I called to make a complaint against a security guard who insulted my cousin calling him fat and to get out of line because he forgot his mask. I was wearing on so asked me to grab product for him then the security guard yelled stepped at me bouncing his chest at me an told me to get out of line as well to join my fat friend"
    sentence = 'Had alot of fun nothing bad to say  Kind of an older crowd in the casino part which is nice'
    #sentence = "Awesome saw Thunder from Down Under. Great show. Friendly Staff. Awesome Casino"
    #base = preprocess(sentence, postag='all')
    #prep_align = alignment(string_indexing(sentence), base)
    #segmented = lda_entropy_segmentation(sentence, list(prep_align), model_1=all_model)
    print(lda_kmeans_entropy_segmentation(sentence, model_1=all_model))
    print(review_segmentation(sentence, model_1=all_model))
    #assert isinstance(segmented, Generator)
    #for item in segmented:
    #    print('*****************************************')
    #    print(item)
    #input('?')
    #sample = pd.read_excel('../data/Canadian_Casinos_preprocessed_corrected.xlsx')[:100]
    #df = segmentation(sample, model_1=all_model)
    #df.to_excel('../data/test_segmentation.xlsx', engine='openpyxl')