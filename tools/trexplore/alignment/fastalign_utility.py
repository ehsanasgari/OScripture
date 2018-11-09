__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu"
__website__ = "https://llp.berkeley.edu/asgari/"


import subprocess
import time
import codecs
import sys
import os
sys.path.append("../")


class FastAlignUtility:
    '''
    This class is python wrapper for running fast align
    '''
    fastalign_path = 'alignment/aligner/fast_align-master/build/'

    def __init__(self):
        '''
        :param language_pair:
        '''

    @staticmethod
    def run_fastalign_file(file, outputdir):
        name = file.split('/')[-1].split('.')[0]
        my_path1=outputdir + name + "_fwd.align"
        my_path2=outputdir + name + "_rev.align"
        if not os.path.exists(my_path1) or os.path.getsize(my_path1) == 0:
            cmd1 = FastAlignUtility.fastalign_path + 'fast_align -I 10 -i ' + file + ' -d -o -v > ' + outputdir + name + "_fwd.align"
            subprocess.getoutput(cmd1)
        if not os.path.exists(my_path2) or os.path.getsize(my_path2) == 0:
            cmd2 = FastAlignUtility.fastalign_path + 'fast_align -I 10 -i ' + file + ' -d -o -v -r > ' + outputdir + name + "_rev.align"
            subprocess.getoutput(cmd2)

    @staticmethod
    def generate_fast_align_files(l1, l1_trans, l2, l2_trans, pair_sentences, lang_file_path):
        my_path=lang_file_path + l1 + '_' + l1_trans + '_' + l2 + '_' + l2_trans + '.txt'
        if not os.path.exists(my_path) or os.path.getsize(my_path) == 0:
            f = codecs.open(lang_file_path + l1 + '_' + l1_trans + '_' + l2 + '_' + l2_trans + '.txt', 'w+', 'utf-8')
            [f.write(' ||| '.join(pair_sentence) + '\n') for pair_sentence in pair_sentences]
            f.close()

    @staticmethod
    def read_fastalign_input(file_address):
        '''
        :param file_address:
        :return: parallel corpora
        '''
        return [[x.strip() for x in line.split(' ||| ')] for line in
                codecs.open(file_address, 'r', 'utf-8').readlines()]

    @staticmethod
    def generate_fast_align_input_from_copora(c1, c2, filename):
        f = codecs.open(filename + '.txt', 'w+', 'utf-8')
        lines=[' ||| '.join([' '.join([x for x in line if not x=='@']), ' '.join([x for x in c2[idx] if not x=='@'])]) for idx, line in enumerate(c1)]
        for line in lines:
            f.write(line+'\n')
        f.close()

    @staticmethod
    def generate_word_alignemnts(par_file, alignment, out_put):
        '''
        :param lang:
        :return: generates tag files mapping from source tokens to targets
        '''
        par_corpus = [line.split(' ||| ') for line in codecs.open(par_file, 'r','utf-8').readlines()]
        alignment = [[(int(pair.split('-')[0]), int(pair.split('-')[1])) for pair in l.split()] for l in
                     codecs.open(alignment,'r', 'utf-8').readlines()]
        f = codecs.open(out_put, 'w','utf-8')
        f.write('\n'.join(
            ['\t'.join(
                [':'.join([e.split()[e_idx], f.split()[f_idx]]) for (e_idx, f_idx) in
                 alignment[k]]) for
             k, (e, f) in
             enumerate(par_corpus)]))
        f.close()
