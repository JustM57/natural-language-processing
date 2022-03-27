from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import pandas as pd


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        c = Counter()
        for text in X:
            c.update(text.split())
        c = c.most_common(self.k)
        self.bow = pd.Series(np.arange(self.k), index=[key for key, val in c])

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        words = Counter(text.split())
        words = pd.Series({key: val for key, val in words.items() if key in self.bow.index})
        idx = self.bow[words.index]
        vector = np.zeros(self.k)
        vector[idx] = words
        return vector

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow.index.tolist()


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        c = Counter()
        for text in X:
            c.update(set(text.split()))
        c = c.most_common(self.k)
        self.idf = OrderedDict({key: (idx, np.log(X.shape[0] / val)) for idx, (key, val) in enumerate(c)})
        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        words = Counter(text.split())
        words = pd.Series({key: val * self.idf[key][1] for key, val in words.items() if key in self.idf.keys()})
        idx = [self.idf[key][0] for key in words.index]
        vector = np.zeros(self.k)
        vector[idx] = words.values
        if self.normalize:
            vector /= (vector.sum() + 1e-8)
        return vector


    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.array([self._text_to_tf_idf(text) for text in X])
