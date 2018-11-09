__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__website__ = "https://llp.berkeley.edu/asgari/"


import codecs
import math
import operator
import numpy as np
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2


class Chi2Analysis(object):
    # X^2 is statistically significant at the p-value level
    def __init__(self, X, Y, feature_names):
        '''
        :param X:
        :param Y:
        :param feature_names:
        '''
        self.X = X
        self.Y = Y
        self.feature_names = feature_names

    def extract_features_fdr(self, file_name, N, alpha=5e-2):
        '''
            Feature extraction with fdr-correction
        '''
        # https://brainder.org/2011/09/05/fdr-corrected-fdr-adjusted-p-values/
        # Filter: Select the p-values for an estimated false discovery rate
        # This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the expected false discovery rate.
        selector = SelectFdr(chi2, alpha=alpha)
        selector.fit_transform(self.X, self.Y)
        scores = {self.feature_names[i]: (s, selector.pvalues_[i]) for i, s in enumerate(list(selector.scores_)) if
                  not math.isnan(s)}
        scores = sorted(scores.items(), key=operator.itemgetter([1][0]), reverse=True)[0:N]
        f = codecs.open(file_name, 'w')
        c_1 = np.sum(self.Y)
        c_0 = len(self.Y) - np.sum(self.Y)
        f.write('\t'.join(['feature', 'score', 'p-value', 'c11', 'c10', 'c01', 'c00']) + '\n')
        self.X = self.X.toarray()
        pos_scores = []
        for w, score in scores:
            feature_array = self.X[:, self.feature_names.index(w)]
            pos = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 1]
            neg = [feature_array[idx] for idx, x in enumerate(self.Y) if x == 0]
            c11 = np.sum(pos)
            c01 = c_1 - c11
            c10 = np.sum(neg)
            c00 = c_0 - c10

            s=score[0]
            if c11 > ((1.0 * c11) * c00 - (c10 * 1.0) * c01):
                s=-s
            s=np.round(s,2)
            if s>0:
                pos_scores.append([str(w), s, score[1], c11, c10, c01, c00])
            f.write('\t'.join([str(w), str(s), str(score[1])] + [str(x) for x in [c11, c10, c01, c00]]) + '\n')
        f.close()
        return pos_scores
