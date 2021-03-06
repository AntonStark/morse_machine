import os
import pandas as pd

from utils.process import CODE

CSV_DIR = './csv_tables'
CSV_TO_READ = os.path.join(CSV_DIR, 'SampleSubmission.csv')
CSV_TO_WRITE = os.path.join(CSV_DIR, 'MorseLabels.csv')

P = 'PREDICTION'


def main():
    with open(CSV_TO_READ, 'r') as crf:
        ss = pd.read_csv(crf, dtype={'ID': int, P: str})
    is_empty = pd.isnull(ss[P])

    # labels = ss[~is_empty][P].to_list()
    # iterable_of_lists = map(lambda s: s.split(' '), map(str.lower, labels))
    # words = functools.reduce(set.union, map(set, iterable_of_lists))

    code = dict(CODE)

    def to_morse(chars: str):
        chars = chars.replace('#', ' ')

        def drop_double_spaces(chars: str):
            while chars.count('  ') > 0:
                chars = chars.replace('  ', ' ')
            return chars
        chars = drop_double_spaces(chars)
        res = '|'.join([code[ch] for ch in chars.lower()])

        def post_repl(chars: str):
            return chars.replace('| |', ' ')
        return post_repl(res)

    rf = ss[~is_empty].copy()
    rf['MORSE'] = rf['PREDICTION'].apply(to_morse)
    rf.rename(columns={'PREDICTION': 'LABEL'}).to_csv(CSV_TO_WRITE, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
