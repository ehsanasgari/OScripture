__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__website__ = "https://llp.berkeley.edu/asgari/"

import sys

sys.path.append("../")

from parallelbible.accessbible import AccessBible
from alignment.fastalign_utility import FastAlignUtility
from multiprocessing import Pool
from tensetagging.tense_utility import preprocess_crs
import os
from utility.file_utility import recursive_glob
import codecs
import random
import itertools
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import scipy as sp


# from wals.multilingutility import MultiLingualUtility



def some_languages():
    files = [('eng', 'newliving'), ('deu', 'newworld'), ('fra', 'newworld'), ('spa', 'newworld'),
             ('lat', 'novavulgata'), ('crs', 'bible'), ('pes', 'newworld'), ('arb', 'newworld'), ('pap', 'newworld')]

    path_to_files = 'multilingual/'
    path_to_lang_files = path_to_files + 'par_file/'
    if not os.path.exists(path_to_lang_files):
        os.makedirs(path_to_lang_files)

    path_to_algn_files = path_to_files + 'algn_file/'
    if not os.path.exists(path_to_algn_files):
        os.makedirs(path_to_algn_files)

    path_to_tagged_files = path_to_files + 'tagged_langs/'
    if not os.path.exists(path_to_tagged_files):
        os.makedirs(path_to_tagged_files)

    path_to_score_files = path_to_files + 'scores/'
    if not os.path.exists(path_to_score_files):
        os.makedirs(path_to_score_files)

    path_to_general_out = path_to_files + 'general_output/'
    if not os.path.exists(path_to_general_out):
        os.makedirs(path_to_general_out)

    access_bible = AccessBible(AccessBible.path)
    files_all = list(
        itertools.chain(*[[(k, x) for x in v] for k, v in access_bible.get_list_of_all_lang_translations().items()]))

    random.shuffle(files_all)
    files = list(set(files_all[0:100] + files))

    language_pairs = list(itertools.product(files, files))

    print(' pairs were generated')

    verse_final = []
    for l1, l2 in language_pairs:
        if not l1 == l2:
            bible1_dic = access_bible.read_subcorpus_newtestament(l1[0], l1[1])
            bible2_dic = access_bible.read_subcorpus_newtestament(l2[0], l2[1])
            l1_verses = list(bible1_dic.keys())
            l2_verses = list(bible2_dic.keys())
            verses = list(set(l1_verses).intersection(l2_verses))
            if len(verse_final) > 0:
                verse_final = set(verse_final).intersection(verses)
            else:
                verse_final = verses
    verse_final = list(verse_final)
    verse_final.sort()

    print(' intersection determined ')
    for l1, l2 in language_pairs:
        if not l1 == l2:
            bible1_dic = access_bible.read_subcorpus_newtestament(l1[0], l1[1])
            bible2_dic = access_bible.read_subcorpus_newtestament(l2[0], l2[1])
            # l1_verses = list(bible1_dic.keys())
            # l2_verses = list(bible2_dic.keys())
            # verses = list(set(l1_verses).intersection(l2_verses))
            # verses.sort()
            FastAlignUtility.generate_fast_align_files(l1[0], l1[1], l2[0], l2[1], [
                (' '.join(bible1_dic[v].split()), ' '.join(bible2_dic[v].split())) for v in verse_final],
                                                       path_to_lang_files)

            bible1_dic = access_bible.read_subcorpus_newtestament(l1[0], l1[1])
            bible2_dic = access_bible.read_subcorpus_newtestament(l2[0], l2[1])
            # l1_verses = list(bible1_dic.keys())
            # l2_verses = list(bible2_dic.keys())
            # verses = list(set(l1_verses).intersection(l2_verses))
            # verses.sort()
            FastAlignUtility.generate_fast_align_files(l2[0], l2[1], l1[0], l1[1], [
                (' '.join(bible2_dic[v].split()), ' '.join(bible1_dic[v].split())) for v in verse_final],
                                                       path_to_lang_files)

    lang_files = recursive_glob(path_to_lang_files, '*' + '.txt')
    pool = Pool(processes=30)
    pool.map(run_fastalign, lang_files)


class Generate_alignments(object):
    def __init__(self):
        path_to_files = '/mounts/data/proj/asgari/superparallelproj/alignment_data/multilingual_alignment/1000_languages/'

        self.path_to_lang_files = path_to_files + 'par_file/'
        if not os.path.exists(self.path_to_lang_files):
            os.makedirs(self.path_to_lang_files)

        self.path_to_algn_files = path_to_files + 'algn_file/'
        if not os.path.exists(self.path_to_algn_files):
            os.makedirs(self.path_to_algn_files)

        self.path_to_tagged_files = path_to_files + 'tagged_langs/'
        if not os.path.exists(self.path_to_tagged_files):
            os.makedirs(self.path_to_tagged_files)

        self.path_to_verse_files = path_to_files + 'verse_langs/'
        if not os.path.exists(self.path_to_verse_files):
            os.makedirs(self.path_to_verse_files)

        self.path_to_score_files = path_to_files + 'scores/'
        if not os.path.exists(self.path_to_score_files):
            os.makedirs(self.path_to_score_files)

        self.path_to_general_out = path_to_files + 'general_output/'
        if not os.path.exists(self.path_to_general_out):
            os.makedirs(self.path_to_general_out)

        self.access_bible = AccessBible(AccessBible.path)
        files_all = list(itertools.chain(
            *[[(k, x) for x in v] for k, v in self.access_bible.get_list_of_all_lang_translations().items()]))

        random.shuffle(files_all)
        files = files_all  # list(set(files_all[0:100]+files))
        self.language_pairs = list(itertools.product(files, files))
        print(' pairs were generated')

    @staticmethod
    def run_fastalign(lang_file, path_to_algn_files='/mounts/data/proj/asgari/superparallelproj/alignment_data/multilingual_alignment/1000_languages/algn_file/', remove=True, generate_intersect=False):
        '''
        :param lang_file:
        :return: run fast alignment
        '''
        print (lang_file)
        FastAlignUtility.run_fastalign_file(lang_file, path_to_algn_files)
        # generate_intersect
        if generate_intersect:
            name = lang_file.split('/')[-1].split('.')[0]
            FastAlignUtility.generate_intersect_alignments(name, path_to_algn_files)
        if remove:
            os.remove(lang_file)

    def run_in_parallel(self):

        #pool = Pool(processes=50)
        #pool.map(self.generate_file, self.language_pairs)

        lang_files = recursive_glob(self.path_to_lang_files, '*' + '.txt')
        pool = Pool(processes=1)
        pool.map(Generate_alignments.run_fastalign, lang_files)

    def generate_file(self, languages):
        l1, l2 = languages
        if not l1 == l2 and not os.path.exists(
                                self.path_to_verse_files + '_'.join(['_'.join(l1), '_'.join(l2)]) + '_verse.txt'):
            bible1_dic = self.access_bible.read_subcorpus_newtestament(l1[0], l1[1])
            bible2_dic = self.access_bible.read_subcorpus_newtestament(l2[0], l2[1])
            l1_verses = list(bible1_dic.keys())
            l2_verses = list(bible2_dic.keys())
            verses = list(set(l1_verses).intersection(l2_verses))
            verses.sort()
            if len(verses) > 100:
                f = codecs.open(self.path_to_verse_files + '_'.join(['_'.join(l1), '_'.join(l2)]) + '_verse.txt', 'w',
                                'utf-8')
                for x in verses:
                    f.write(x + '\n')
                f.close()

                FastAlignUtility.generate_fast_align_files(l1[0], l1[1], l2[0], l2[1], [
                    (' '.join(bible1_dic[v].split()), ' '.join(bible2_dic[v].split())) for v in verses],
                                                           self.path_to_lang_files)


if __name__ == '__main__':
    # files=[('eng','newliving'),('deu','newworld'),('fra','newworld'),('spa','newworld'),('lat','novavulgata'),('crs','bible'),('pes','newworld'),('arb','newworld'),('pap','newworld')]
    GA = Generate_alignments()
    GA.run_in_parallel()
