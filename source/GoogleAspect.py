import pickle
import gensim
import numpy as np
import pandas as pd
global aspect
google_aspect = [['fitness', 'golf', 'play', 'tennis', 'basketball', 'horse', 'ride', 'beach'],
                 ['family', 'kid', 'group', 'children', 'child', 'friend'],
                 ['nature', 'scenery', 'scene', 'view', 'view', 'view', 'city', 'resort', 'lake', 'outdoors', 'site', 'area', 'space', 'area'],
                 ['amenities', 'breakfast', 'floor', 'place', 'bedroom', 'room', 'setting', 'bed', 'accommodation', 'suite',
                  'property', 'hotel'],
                 ['food', 'dinner', 'patio', 'restaurant', 'buffet', 'soup', 'lamb', 'seafood', 'rib', 'meal',
                  'dining', 'lobster'],
                 ['staff', 'service', 'service', 'staff', 'team', 'manager', 'employee', 'people', 'people', 'people', 'desk'],
                 ['nightlife', 'music', 'club', 'dance', 'dancing', 'music', 'show', 'concert', 'firework', 'entertainment','pub'],
                 ['cleanliness', 'clean', 'comfortable', 'mask', 'covid', 'pandemic', 'spot', 'stain', 'room', 'standard'],
                 ['couples', 'wedding', 'couples', 'anniversary', 'romantic', 'wedding', 'husband', 'wife', 'bride', 'honeymoon'],
                 ['business', 'conference', 'business', 'conference', 'event', 'hall', 'summit', 'meeting', 'company', 'corporate',
                  'venue', 'venue'],
                 ['spa\\pool', 'wellness', 'massages', 'pedicures', 'spa', 'spa', 'health', 'health', 'health', 'steam',
                  'spa', 'pool', 'swim', 'swimming', 'whirlpool', 'waterpark', 'pool', 'water', 'pool', 'pool'],
                 ['bathroom', 'shower', 'tub', 'towel', 'sofa', 'laundry', 'toilet', 'washroom', 'soap', 'bathtub', 'dishwasher',
                  'shampoo'],
                 ['bar', 'drink', 'drink', 'drink', 'beer', 'tea', 'bar', 'bar', 'bar'],
                 ['kitchen', 'menu', 'waitress', 'waiter', 'restaurant'],
                 ['location', 'building', 'station', 'lobby'],
                 ['atmosphere', 'ambiance', 'city', 'atmosphere', 'atmosphere'],
                 ['parking', 'parking', 'car', 'entrance', 'parking', 'road'],
                 ['accessibility', 'disable', 'elevators', 'trolley', 'stair', 'accessible', 'path', 'walkways', 'access'],
                 ['entertainment', "venue", "concert", "concert", "entertainment", "sound", "system", "theatre", "performance"],
                 ['safety\\security', 'safety', 'precaution', 'security', 'security', 'guard', 'door'],
                 ['wi-fi', 'internet', 'cable', 'wireless'],
                 ['slot machine', 'machine', 'machine', 'machine', 'slot'],
                 ['poker', 'poker', 'play', 'dealer', 'card', 'player', 'hand', 'card'],
                 ['customer service', 'customer', 'customer', 'person', 'manager', 'care'],
                 ['experience', 'experience', 'experience'],
                 ['money', 'money', 'money', 'win', 'win', 'cash', 'chance'],
                 ['price', 'price', 'price', 'quality'],
                 ['gamble', 'table', 'table', 'roulette', 'blackjack', 'game', 'table', 'bet']]


def inference(lda, corpus_element):
    return [lda[corpus_element][j][1] for j in range(lda.num_topics)]


def inference_distribution_matrix(lda, corpus):
    for line in corpus:
        yield inference(lda, line)


def community_detection(lda, corpus):
    # Inference Distribution Matrix --> idm
    idm = np.array(list(inference_distribution_matrix(lda, corpus))).transpose()
    return [idm[j].argmax() if idm.transpose()[idm[j].argmax()].argmax() == j else -1 for j in range(0, lda.num_topics)]


if __name__ == '__main__':
    pxp_model = pickle.load(open('../pipeline/pxp_model_noun.pxp', 'rb'))
    pxp_model.lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(pxp_model.lda_model, gamma_threshold=0.001,
                                                                iterations=50)
    
    google_corpus = [pxp_model.dictionary.doc2bow(text) for text in aspect]
    google_label = community_detection(pxp_model.lda_model, google_corpus)
    print(google_label)
    print([aspect[google_label[ind]][0] if google_label[ind] != -1 else '*' for ind in
           range(0, len(google_label))])

