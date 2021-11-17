import gensim
import numpy as np
from source.LDAInference import scissors, lda_mount
import pickle
import pandas as pd
from source.AspectOpinionOccurrence import lda_occurrence, represent, SentimentOccurrence, precision, interquartile, \
    topic_outlier, spectral_highlight


class TopicNode:
    def __init__(self, text:str, score:float, label_doc:np.asarray):
        self.text = text
        self.score = score
        self.label = self.__update_label(label_doc)
    
    def __add__(self, other):
        if not isinstance(other, TopicNode): return self
        if self.label is not other.label: return self
        return TopicNode(self.text, self.score + other.score, self.label)
    
    def __str__(self) -> str:
        return str(self.text) + "|" + str(self.score) + "|" + str(self.label)
    
    def __update_label(self, label_doc:np.asarray):
        try:    return [label_doc[i][0] for i in range(len(label_doc)) if self.text in label_doc[i]][0]
        except: return None


class TopicCluster(list):
    def __init__(self, stream:str, label_doc:np.asarray):
        super().__init__()
        self.label_doc = label_doc
        f_part, s_part = scissors(stream)
        for i in range(len(f_part)):
            self.append(TopicNode(text=s_part[i], score=f_part[i], label_doc=label_doc))
    
    def __str(self):
        for item in self:
            yield str(item)
            
    def __str__(self):
        return " -- ".join(list(self.__str()))
    
    def clustering(self):
        label_set = [item.label for item in self if item.label is not None]
        label_dict = {label_set[i]:np.array([node for node in self if node.label is label_set[i]]).sum().score
                      for i in range(0, len(label_set))}
        return label_dict

    
def occ_scatter(cell, pxp_model, label_doc:np.asarray, row_index:int) -> pd.DataFrame:
    scatter_dict = TopicCluster(pxp_model.lda_model.print_topic(row_index), label_doc).clustering()
    dictionary_scaling = lambda dictionary, scalar: {list(dictionary.keys())[i]: scalar * list(dictionary.values())[i]
                                                     for i in range(0, len(dictionary))}
    
    return dictionary_scaling(scatter_dict, cell)


def ao_cluster(aspect_model, opinion_model, parent_document:pd.DataFrame, label_doc:np.asarray):
    ao_pair = lda_occurrence(aspect_model, opinion_model, parent_document)

    ind = [label_doc[i][0] for i in range(0, len(label_doc))]
    col = represent(opinion_model)
    
    dataframe = pd.DataFrame(np.array([[SentimentOccurrence(0, 0) for __ in range(0, len(ao_pair[0]))] for _ in range(0, len(ao_pair))]), index=ind, columns=col)
    for i in range(0, len(ao_pair)):
        for j in range(0, len(ao_pair[0])):
            map_image = occ_scatter(ao_pair[i][j], pxp_model=aspect_model, label_doc=label_doc, row_index=i)
            for key, score in map_image.items():
                dataframe.loc[key][j] += score
    
    dataframe = interquartile(topic_outlier(dataframe))
    print(dataframe)
    return dataframe.style.applymap(precision).\
                     applymap(spectral_highlight)
                     #to_excel(save_path)
              

if __name__ == '__main__':
    pxp_noun = lda_mount('../pipeline/pxp_model_flair_aspect_LCBO.pxp')
    
    pxp_adj = lda_mount('../pipeline/pxp_model_flair_opinion_LCBO.pxp')
    
    
    print(pxp_noun.lda_model.print_topic(4))
    a = TopicCluster(pxp_noun.lda_model.print_topic(0), pd.read_excel('../arash_labels.xlsx', header=None).transpose().to_numpy(na_value=""))
    print(a)
    for key, item in a.clustering().items():
        print(key, item)
    
    x = ao_cluster(pxp_noun, pxp_adj, pd.read_excel("../data/LCBO_preprocessed_corrected.xlsx"),
                   pd.read_excel('../arash_labels.xlsx', header=None).transpose().to_numpy(na_value=""))
    x.to_excel("../pipeline/AOocc_LCBO3_report_scattering.xlsx")