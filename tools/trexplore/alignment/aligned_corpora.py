__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__website__ = "https://llp.berkeley.edu/asgari/"

import itertools
import codecs
import sys

sys.path.append("../")

class AlignedCorpora:
    '''
    This class works with multiple aligned
    '''

    def __init__(self, parallel_dict):
        '''
        :param parallel_dict: a dict of aligned sentences (verses)
        '''
        self.langs = list(parallel_dict.keys())
        self.langs.sort()
        self.parallel_dict = parallel_dict

    def generate_fastalign_output(self, output_dir):
        '''
        :param output_dir: directory to generate output files
        :return:  language pair list
        '''
        all_lang_pairs = list(itertools.combinations(self.langs, 2))
        for (l1, l2) in all_lang_pairs:
            print('creating ' + output_dir + l1 + '_' + l2 + '.txt ..')
            f = codecs.open(output_dir + l1 + '_' + l2 + '.txt', 'w', 'utf-8')
            num_verses = len(self.parallel_dict[l1])
            for idx in range(0, num_verses):
                f.write(self.parallel_dict[l1][idx] + ' ||| ' + self.parallel_dict[l2][idx] + '\n')
            f.close()
        return all_lang_pairs
