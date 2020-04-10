# Adapted from https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py

import argparse
import gzip
import io
import os
import subprocess

from collections import Counter
from urllib import request


def main(lm_path, top_n_words=None):
    dir = '.'

    # Grab corpus.
    data_dir = os.path.join(dir, 'openwebtext')
    if not os.path.isdir(data_dir):
        raise Exception('missing data')

    # # Convert to lowercase and count word occurences.
    # counter = Counter()
    # data_lower = os.path.join(dir, 'lower.txt.gz')
    # print('Converting to lower case and counting word frequencies...')
    # with io.TextIOWrapper(io.BufferedWriter(gzip.open(data_lower, 'w')), encoding='utf-8') as lower:
    #     with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
    #         for line in upper:
    #             line_lower = line.lower()
    #             counter.update(line_lower.split())
    #             lower.write(line_lower)

    # # Build pruned LM.
    # order = 3
    # prune_args = '0 12 24'.split()
    # assert len(prune_args) <= order

    # lm_path = os.path.join(dir, 'lm-{}gram-{}.arpa'.format(order, '-'.join(prune_args)))
    # print('Creating ARPA file...')
    # subprocess.check_call([
    #     './bin/lmplz', '--order', str(order),
    #     '--temp_prefix', dir,
    #     '--memory', '50%',
    #     '--text', data_lower,
    #     '--arpa', lm_path,
    #     '--prune'] + prune_args
    # )

    # subprocess.check_call(['xz', '-v', '-f', '-k', lm_path])

    with open(lm_path, 'r', encoding='utf-8') as arpa_file:
        unigrams = dict()
        in_1_grams = False
        for line in arpa_file:
            if not in_1_grams and line.startswith(r'\1-grams:'):
                in_1_grams = True
                continue
            if in_1_grams and line.startswith(r'\2-grams:'):
                break
            tokens = line.split()
            if in_1_grams and tokens:
                log_probability, word, back_off_weight = tokens
                unigrams[word] = float(log_probability)
    most_common_words = sorted(unigrams.items(), key=lambda pair: pair[1], reverse=True)

    if top_n_words:
        # Filter LM using vocabulary of top N words
        filtered_path = os.path.join(dir, lm_path.replace('.arpa', '-top%d.arpa' % top_n_words))
        vocab_str = '\n'.join(
            word for word, log_probability in most_common_words[:top_n_words])
        print('Filtering ARPA file...')
        subprocess.run(['./bin/filter', 'single', 'model:{}'.format(lm_path), filtered_path], input=vocab_str.encode('utf-8'), check=True)
        subprocess.check_call(['xz', '-v', '-f', '-k', filtered_path])
    else:
        filtered_path = lm_path

    # # Quantize and produce trie binary.
    # print('Building lm.binary...')
    # subprocess.check_call([
    #     './bin/build_binary', '-a', '255',
    #     '-q', '8',
    #     'trie',
    #     filtered_path,
    #     'lm.binary'
    # ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lm_path')
    parser.add_argument('-w', '--topwords', type=int, default=500000, help="top number of words to filter LM to")
    args = parser.parse_args()
    main(top_n_words=args.topwords, lm_path=args.lm_path)
