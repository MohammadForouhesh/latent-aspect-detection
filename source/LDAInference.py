import pickle
import gensim
from source import LDATopicModeling
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn


def scissors(stream:str) -> (list, list):
    word_list = stream.replace("*", " ").replace(" +", "").split(" ")
    float_part = []
    string_part = []
    for item in word_list:
        try:
            f_razor = float(item)
            float_part.append(f_razor)
        except:
            if isinstance(item, str):
                string_part.append(item.replace('"', ''))
    return float_part, string_part


def lda_mount(path:str) -> LDATopicModeling.TopicModeling:
    model = pickle.load(open(path, 'rb'))
    model.lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model.lda_model, gamma_threshold=0.001,
                                                                            iterations=50)
    
    return model


def lda_inference(model:LDATopicModeling.TopicModeling, preprocessed_sentence:str) -> list:
    corpus = [model.dictionary.doc2bow(text) for text in [preprocessed_sentence.split(" ")]]
    return [model.lda_model[corpus[0]][j][1] for j in range(model.lda_model.num_topics)]


def topic_detection(preprocessed_sentence:str, model: LDATopicModeling.TopicModeling=None):
    lda_distro = lda_inference(model, preprocessed_sentence)
    return np.argsort(lda_distro)[::-1]


def resnik_similarity(word:str, other_words:str, corpus_ic:dict) -> float:
    simil = []
    for other_word in other_words.split(' '):
        word_syn = wn.synsets(word, pos=['v', 'n'])
        other_syn = wn.synsets(other_word, pos=['v', 'n'])
        resnik_simil_score = [word_sense.res_similarity(other_sense, corpus_ic)
                              for word_sense in word_syn for other_sense in other_syn
                              if word_sense.pos() == other_sense.pos()]

        simil.append(max(resnik_simil_score) if len(resnik_simil_score) != 0 else 0)

    return sum(simil)


def word_topic_score(word:str, model:LDATopicModeling.TopicModeling, corpus_ic:dict):
    topic_list = model.lda_model.print_topics(-1)
    for ind, stream in topic_list:
        f_vec, s_vec = scissors(stream)
        scaler = MinMaxScaler()
        f_normal = list(scaler.fit_transform(np.array(f_vec).reshape(-1, 1)).reshape(-1))
        score = max([f_normal[i]*resnik_similarity(s_vec[i], word, corpus_ic) for i in range(0, len(s_vec))])
        yield ind, score


def word_topic_detection(word:str, model:LDATopicModeling.TopicModeling, corpus_ic:dict):
    score_list = list(word_topic_score(word, model, corpus_ic))
    return max(score_list, key=lambda item: item[1])


if __name__ == '__main__':
    from source.Preprocessing import preprocess
    
    test_model = lda_mount('../pipeline/pxp_model_flairaspect.pxp')
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    print('const: ', 1.3 / test_model.lda_model.num_topics)
    sentence = 'The sake menu should not be overlooked'
    prep = preprocess(sentence, postag=None, tagger=None)
    print(prep)
    eval_distro = lda_inference(test_model, prep)
    print(np.array(eval_distro).argmax(), max(eval_distro), test_model.lda_model.print_topic(np.array(eval_distro).argmax()))
    topic = word_topic_detection('food', test_model, brown_ic)
    print('1', topic, test_model.lda_model.print_topic(topic[0]))
    
    sentence = 'good music , great food , speedy service affordable prices'
    prep = preprocess(sentence, postag=None, tagger=None)
    print(prep)
    eval_distro = lda_inference(test_model, prep)
    print(np.array(eval_distro).argmax(), max(eval_distro), test_model.lda_model.print_topic(np.array(eval_distro).argmax()))
    topic = word_topic_detection('afternoon', test_model, brown_ic)
    print('2', topic, test_model.lda_model.print_topic(topic[0]))
    
    sentence = 'otherwise , good stuff for late nite eats'
    prep = preprocess(sentence, postag=None, tagger=None)
    print(prep)
    eval_distro = lda_inference(test_model, prep)
    print(np.array(eval_distro).argmax(), max(eval_distro), test_model.lda_model.print_topic(np.array(eval_distro).argmax()))
    topic = word_topic_detection('place', test_model, brown_ic)
    print('3', topic, test_model.lda_model.print_topic(topic[0]))
