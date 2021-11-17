import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from source.Preprocessing import nlp
from source.eval.Evaluation import pytrec_evaluation


class AspectKMeans:
    def __init__(self, num_clusters: int):
        self.vectorizer = TfidfVectorizer()
        self.num_clusters = num_clusters
        self.km = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=1)

    def __prep(self, text: str):
        item = text.replace("!", ". ").replace(".", ". ").replace("?", ". ")
        return ' '.join([sent for sent in item.split(' ')])

    def train(self, train_set: pd.DataFrame):
        prep_train = train_set.apply(self.__prep)
        X = self.vectorizer.fit_transform(prep_train)
        self.km.fit(X)
        return self.km

    def inference(self, text: str):
        Y = self.vectorizer.transform([self.__prep(text)])
        pred = self.km.predict(Y)
        return list(pred)


def kmeans_eval_indexer(aspects, model: AspectKMeans):
    Y = model.vectorizer.transform(aspects)
    return list(model.km.predict(Y))


def akmeans_evaluation_functional(evaluation_set: pd.DataFrame, model: AspectKMeans):
    evaluation_set['detected_aspect'] = evaluation_set['caption'].apply(model.inference)
    return pytrec_evaluation(evaluation_set['sam_eval_aspect'].apply(kmeans_eval_indexer, model=model),
                             evaluation_set['detected_aspect'], success_at=100) \
        .to_frame() \
        .transpose()

