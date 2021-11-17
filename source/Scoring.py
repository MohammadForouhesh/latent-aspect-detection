import numpy as np
import pandas as pd
from textblob import TextBlob

pxp_score_scale = lambda inverse_polarity: 'sad' if inverse_polarity <= 6 else 'happy' if 8 < inverse_polarity else 'meh'

polarity_scale = lambda pxp_score: (pxp_score - 5)/5
inverse_polarity_scale = lambda polarity: (polarity + 1) * 5

segment_polarity_map = lambda text_list: [item.sentiment.polarity for item in (map(TextBlob, text_list))]

happy_meh_sad_functional = lambda polarity_list: pd.Series(list(map(pxp_score_scale, map(inverse_polarity_scale, polarity_list)))).value_counts()


def pxp_scoring(happy=0, meh=0, sad=0):
    const = 10/(1.57 + 1.10714872)
    score = happy*5 if (meh*2)+(sad*8) < 1 else (happy*5)/((meh*2)+(sad*8)+0.001)
    result = np.arctan(score - 2) - np.pi/2
    return const*result + 10


def review_scoring(segment_list):
    series = happy_meh_sad_functional(segment_polarity_map(segment_list))
    return polarity_scale(pxp_scoring(**dict(series)))


if __name__ == '__main__':
    data = ['I am glad that I visited this wonderful place',
            'I am sure this will be of much help',
            'of line as well to join my fat friend',
            'to grab product for him then the security guard yelled stepped at me bouncing his chest at me an told me to get out',
            'who insulted my cousin calling him fat and to get out of line because he forgot his mask. I was wearing on so asked me',
            'I called today got no answer and the machines were full so hung up on. I called to make a complaint against a security guard',
            "I called today got no answer and the machines were full so hung up on. I called to make a complaint against a security guard who insulted my cousin calling him fat and to get out of line because he forgot his mask. I was wearing on so asked me to grab product for him then the security guard yelled stepped at me bouncing his chest at me an told me to get out of line as well to join my fat friend"]

    print(review_scoring(data))
