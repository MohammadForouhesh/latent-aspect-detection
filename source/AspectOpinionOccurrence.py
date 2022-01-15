import pickle
import datetime

import gensim
import colorsys
import numpy as np
import pandas as pd
from numpy import percentile
from textblob import TextBlob
import gensim.downloader as api
from scipy.spatial import distance
from source.GoogleAspect import community_detection, google_aspect
from source.LDAInference import scissors, lda_inference

# =====================================================PANDAS=STYLE======================================================
from source.Logging import process_logger
from source.Preprocessing import preprocess
from source.Scoring import review_scoring
from source.Segmentation import lda_kmeans_entropy_segmentation

pxp_scoring_system = lambda item: (item - 0.6)*2.5 if 0.6 <= item <= 1 else 0 if 0 <= item < 0.6 else item


class SentimentOccurrence():
    def __init__(self, occurrence, polarity):
        self.occurrence = occurrence
        self.polarity = polarity
    
    def __str__(self):
        return str(self.occurrence)
    
    def __repr__(self):
        return str(self.occurrence)
    
    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return SentimentOccurrence(self.occurrence + other.occurrence, self.polarity + other.polarity)
    
    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return SentimentOccurrence(self.occurrence - other.occurrence, self.polarity - other.polarity)
    
    def __mul__(self, other):
        return SentimentOccurrence(self.occurrence * other, self.polarity * other)
    
    def __rmul__(self, other):
        return SentimentOccurrence(self.occurrence * other, self.polarity * other)
    
    def __div__(self, other):
        return SentimentOccurrence(self.occurrence / other, self.polarity)
    
    def __truediv__(self, other):
        return SentimentOccurrence(self.occurrence / other, self.polarity)
    
    def __lt__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return self.occurrence < other.occurrence
    
    def __le__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return self.occurrence <= other.occurrence
    
    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return self.occurrence == other.occurrence
    
    def __ne__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return self.occurrence != other.occurrence
    
    def __ge__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return self.occurrence >= other.occurrence
    
    def __gt__(self, other):
        if isinstance(other, int) or isinstance(other, float): other = SentimentOccurrence(other, 0)
        return self.occurrence > other.occurrence


def to_hex(i):
    hx = hex(int(i))[2:]
    if len(hx) == 1:
        hx = "0" + hx
    return hx


def css_beauty(h, polarity, s) -> str:
    if polarity == 0:           return 'background-color: white'
    if abs(polarity) >= 0.8:   polarity = (polarity/abs(polarity))*0.8
    r, g, b = colorsys.hls_to_rgb(h, 1 - abs(polarity), s)
    cs = [to_hex(c) for c in [r * 255, g * 255, b * 255]]
    param = "#" + "".join(cs)
    string = 'background-color: ' + param
    if 1 - abs(polarity) < 0.3125:
        string = string + '; color:white'
    return string


def spectral_highlight(text):
    if text.polarity > 0.15:
        h = 0.3209459459459459
        # l = green spectrum
        s = 0.8604651162790697
        return css_beauty(h, text.polarity, s)
    
    elif text.polarity < 0:
        h = 0
        # l = red spectrum
        s = 0.9458333333333333
        return css_beauty(h, text.polarity, s)
    
    else:
        h = 0.175
        # l = yellow luminosity
        s = 0.8604651162790697
        return css_beauty(h, text.polarity, s)


def precision(text):
    text.occurrence = round(text.occurrence, 3)


# =====================================================VALID=PAIR=======================================================
def valid_pair(dataframe:pd.DataFrame) -> pd.DataFrame:
    validity = [["" for _ in range(0, dataframe.shape[1])] for __ in range(0, dataframe.shape[0])]
    similar_ = [[] for __ in range(0, dataframe.shape[0])]
    for ind_aspect in range(0, dataframe.shape[0]):
        row_vec = model.get_vector(dataframe.index[ind_aspect])
        for ind_opinion in range(0, dataframe.shape[1]):
            col_vec = model.get_vector(dataframe.columns[ind_opinion])
            similar_[ind_aspect].append((1 - distance.cosine(row_vec, col_vec)).round(4))
    indicator = [min(percentile(similar_[i], 80), 0.8) for i in range(0, dataframe.shape[0])]
    print('indicator: ', indicator)
    for ind_aspect in range(0, dataframe.shape[0]):
        for ind_opinion in range(0, dataframe.shape[1]):
            if similar_[ind_aspect][ind_opinion] >= indicator[ind_aspect]: continue
            validity[ind_aspect][ind_opinion] = 'background-color: gray; color: gray'
    
    return pd.DataFrame(np.array(validity), index=dataframe.index, columns=dataframe.columns)


# ================================================INTERQUARTILE=OUTLIER==================================================
def interquartile(dataframe:pd.DataFrame) -> pd.DataFrame:
    data = [[] for _ in range(dataframe.shape[0])]
    for i in range(dataframe.shape[0]):
        for j in range(dataframe.shape[1]):
            data[i].append(dataframe.iloc[i][j])
    
    __temp = [item.occurrence for row in data for item in row]
    max_ = max(__temp)
    min_ = min(__temp)
    print(np.array(__temp).mean(), np.array(__temp).std())
    print(min_, max_)
    
    q25, q75 = percentile(__temp, 25), percentile(__temp, 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    
    cut_off = 2
    upper = q75 + iqr * cut_off #np.array(sorted(__temp, reverse=True)[:20]).mean)
    lower = max(q25 - iqr * cut_off, percentile(__temp, 15))
    print(lower, upper)
    
    interquart = [[] for _ in range(0, dataframe.shape[0])]
    for i in range(0, dataframe.shape[0]):
        for item in data[i]:
            val = item if lower <= item <= upper else SentimentOccurrence(upper, item.polarity) \
                if item > upper else SentimentOccurrence(lower, item.polarity)
            interquart[i].append(val)
    
    rescaled = np.array([np.array([(1-0)/(upper-lower)*(value-lower) for value in row]) for row in interquart])
    return pd.DataFrame(rescaled, index=dataframe.index, columns=dataframe.columns)


# ====================================================TOPIC=OUTLIER======================================================
def topic_outlier(dataframe:pd.DataFrame) -> pd.DataFrame:
    index_dispose = [item for item in dataframe.index if '*' in item]
    column_dispose = []

    data = [[] for _ in range(dataframe.shape[0])]
    for i in range(dataframe.shape[0]):
        for j in range(dataframe.shape[1]):
            data[i].append(dataframe.iloc[i][j])
            
    temp_mean = np.array([item.occurrence for row in data for item in row]).mean()
    """
    for ind_aspect in range(0, dataframe.shape[0]):
        if np.array(sorted(dataframe.iloc[ind_aspect], reverse=True)[:dataframe.shape[0]//5]).mean() < temp_mean:
            index_dispose.append(dataframe.index[ind_aspect])
    """
    for ind_opinion in range(0, dataframe.shape[1]):
        if np.array(sorted(dataframe.iloc[:, ind_opinion], reverse=True)[:dataframe.shape[1]//5]).mean() < temp_mean:
            column_dispose.append(dataframe.columns[ind_opinion])

    return dataframe.drop(index=index_dispose, columns=column_dispose)


# =======================================================================================================================
def lda_distribution_normalizer(lda_distribution:list, num_topics:int) -> np.ndarray:
    p_value = round(1/num_topics, 2) + 0.01
    # p_value = 0.035
    topic_vec = [lda_distribution[j] if lda_distribution[j] > p_value else 1e-8 for j in range(0, num_topics)]
    return np.array(topic_vec) * (1/max(max(topic_vec), p_value))


def represent(pxp_model) -> list:
    rep = []
    for id in range(0, pxp_model.lda_model.num_topics):
        _, string_part = scissors(pxp_model.lda_model.print_topic(id))
        rep.append(string_part[0])
    return rep


def pxp_labeling(pxp_model, labeling_doc=google_aspect) -> list:
    labeling_corpus = [pxp_model.dictionary.doc2bow(text) for text in labeling_doc if text is not np.nan]
    labeling_community = community_detection(pxp_model.lda_model, labeling_corpus)
    representative = represent(pxp_model)
    return [labeling_doc[labeling_community[ind]][0] if labeling_community[ind] != -1 else representative[ind] + "*" for ind in
            range(0, len(labeling_community))]


# ======================================================================================================================
def lda_occurrence(aspect_model, opinion_model,  parent_document:pd.DataFrame) -> list:
    num_topics_opinion = opinion_model.lda_model.num_topics
    num_topics_aspect = aspect_model.lda_model.num_topics
    
    occurrence_matrix = np.array([[SentimentOccurrence(0, 0) for _ in range(0, num_topics_opinion)]\
                                  for __ in range(0, num_topics_aspect)])
    
    for i in range(0, len(parent_document)):
        sentence = parent_document['caption'].iloc[i]
        preprocessed_sentence = parent_document['all_preprocessed'].iloc[i]
        aspect_dist = lda_distribution_normalizer(lda_inference(aspect_model, preprocessed_sentence), num_topics_aspect)
        opinion_dist = lda_distribution_normalizer(lda_inference(opinion_model, preprocessed_sentence), num_topics_opinion)
        
        # for sentence in segmented_doc.caption:
        if isinstance(sentence, float) or isinstance(sentence, int): print(sentence); continue
        sentence = sentence.replace('.', ' ')
        pol = TextBlob(sentence).sentiment.polarity
        occ_dist = aspect_dist.reshape(-1, 1)*opinion_dist
        sent_dist = pol*occ_dist
        obj_dist = np.array([[SentimentOccurrence(occ_dist[row][col], sent_dist[row][col]) for col in\
                            range(0, occ_dist.shape[1])] for row in range(0, occ_dist.shape[0])])
        occurrence_matrix += obj_dist
            
    return occurrence_matrix


def topic_relevance(aspect_model, opinion_model, parent_document:pd.DataFrame, save_path, **kwargs) -> list:

    excel_array = [[], [], [], [], [], [], [], [], []]

    num_topics_opinion = opinion_model.lda_model.num_topics
    num_topics_aspect = aspect_model.lda_model.num_topics

    for i in range(0, len(parent_document)):
        sentence = parent_document['caption'].iloc[i]
        preprocessed_sentence = parent_document['all_preprocessed'].iloc[i]
        aspect = parent_document['aspect_preprocessed'].iloc[i]
        opinion = parent_document['opinion_preprocessed'].iloc[i]
        aspect_dist = lda_distribution_normalizer(lda_inference(aspect_model, preprocessed_sentence), num_topics_aspect)
        opinion_dist = lda_distribution_normalizer(lda_inference(opinion_model, preprocessed_sentence), num_topics_opinion)
        
        aspect_id = list(aspect_dist).index(max(aspect_dist))
        opinion_id = list(opinion_dist).index(max(opinion_dist))
        
        pol = TextBlob(sentence).sentiment.polarity
        index = parent_document.index[i]
        excel_array[0].append(index)
        excel_array[1].append(aspect_model.lda_model.print_topic(aspect_id))
        excel_array[2].append(opinion_model.lda_model.print_topic(opinion_id))
        excel_array[3].append(aspect)
        excel_array[4].append(opinion)
        excel_array[5].append(sentence)
        excel_array[6].append(pol)
        excel_array[7].append(max(aspect_dist))
        excel_array[8].append(max(opinion_dist))
        
    nparray = np.array(excel_array).transpose()
    temp_dataframe = pd.DataFrame(nparray, columns=["index", "aspect", "opinion", "preprocessed_aspect", "preprocessed_opinion",
                                                    "caption", "sentiment", "dist_aspect", "dist_opinion"])
    temp_dataframe.to_excel(save_path)
    return temp_dataframe


def occurrence_builder(aspect_model, opinion_model, all_model, parent_document:pd.DataFrame, save_path:str,
                       labeling_doc=google_aspect):
    
    if labeling_doc is None: labeling_doc = google_aspect
    
    occurrence_matrix = lda_occurrence(aspect_model, opinion_model, all_model, parent_document)
    
    #row = represent(aspect_model)
    col = represent(opinion_model)
    row = pxp_labeling(aspect_model, labeling_doc=labeling_doc)
    # col = google_labeling(opinion_model)
    
    temp_dataframe = pd.DataFrame(np.array(occurrence_matrix), index=row, columns=col)
    temp_dataframe = temp_dataframe.groupby(by=temp_dataframe.columns, axis=1).sum()
    temp_dataframe = temp_dataframe.groupby(by=temp_dataframe.index, axis=0).sum()
    
    __temp = interquartile(topic_outlier(temp_dataframe))
    __temp.style.applymap(precision).\
        applymap(spectral_highlight).\
        to_excel(save_path)
        #apply(valid_pair, axis=None).\
    

if __name__ == '__main__':
    # model_wiki = api.load("glove-wiki-gigaword-50")
    # model_ggl_news = api.load("word2vec-google-news-300")
    # model_twitter_huge = api.load("glove-twitter-200")
    model_twitter_small = api.load("glove-twitter-25")
    global model
    model = model_twitter_small
    
    pxp_model_opinion = pickle.load(open('../pipeline/pxp_model_flair_opinion_t1_preprocessed_corrected.pxp', 'rb'))
    lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(pxp_model_opinion.lda_model, gamma_threshold=0.001, iterations=50)
    pxp_model_opinion.lda_model = lda
    
    pxp_model_aspect = pickle.load(open('../pipeline/pxp_model_flair_aspect_t1_preprocessed_corrected.pxp', 'rb'))
    lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(pxp_model_aspect.lda_model, gamma_threshold=0.001, iterations=50)
    pxp_model_aspect.lda_model = lda
    path = '../data/t1_preprocessed_corrected_preprocessed_corrected_all_segmented.xlsx'
    doc = pd.read_excel(path)
    occurrence_builder(pxp_model_aspect, pxp_model_opinion, parent_document=doc, save_path='../results/occurrence_lcbo.xlsx',
                       labeling_doc=pd.read_excel('../labels/arash_labels.xlsx', header=None).transpose().to_numpy(na_value=""))
    topic_relevance(pxp_model_aspect, pxp_model_opinion, parent_document=doc, save_path='results/occurrence_relevance_aspect_opinion_lcbo.xlsx')
    """
    path = '../data/test_database.xlsx'
    aspect_document = pd.read_excel('data/segmentation/aspect_test.xlsx', index_col=0)
    series = np.asarray(aspect_document["preprocessed"])
    string = [str(line).split() for line in series if line is not np.nan]
    aspect_corpus = [pxp_model_aspect.dictionary.doc2bow(text) for text in string]
    
    opinion_document = pd.read_excel('data/segmentation/aspect_test.xlsx', index_col=0)
    series = np.asarray(opinion_document["preprocessed"])
    string = [str(line).split() for line in series if line is not np.nan]
    opinion_corpus = [pxp_model_opinion.dictionary.doc2bow(text) for text in string]

    occurrence_builder(pxp_model_aspect, pxp_model_opinion, path, save_path='test/inference_occurrence17.xlsx',
                       aspect_corpus=aspect_corpus, opinion_corpus=opinion_corpus, update=True)
                       """