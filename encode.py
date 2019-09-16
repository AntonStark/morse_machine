# import functools
# import numpy as np
import pandas as pd

CSV_TO_READ = './SampleSubmission.csv'
CSV_TO_WRITE = './MorseLabels.csv'
P = 'PREDICTION'

CODE = [
    ('a', '._'), ('b', '_...'), ('c', '_._.'), ('d', '._.'), ('e', '.'), ('f', '.._.'), ('g', '__.'), ('h', '....'),
    ('i', '..'), ('j', '.___'), ('k', '_._'), ('l', '._..'), ('m', '__'), ('n', '_.'), ('o', '___'), ('p', '.__.'),
    ('q', '__._'), ('r', '._.'), ('s', '...'), ('t', '_'), ('u', '.._'), ('v', '..._'), ('w', '.__'), ('x', '_.._'),
    ('y', '_.__'), ('z', '__..'), (' ', ' '),
    ('0', '_____'), ('1', '.____'), ('2', '..___'), ('3', '...__'), ('4', '...._'),
    ('5', '.....'), ('6', '_....'), ('7', '__...'), ('8', '___..'), ('9', '____.'),
    ('~', '._._'), ('<AS>', '._...'), ('<AR>', '._._.'), ('<SK>', '..._._'),
    ('<KN>', '_.__.'), ('<INT>', '.._._'), ('<HM>', '....__'), ('<VE>', '..._.'),
    ('\\', '._.._.'), ('\'', '.____.'), ('$', '..._.._'), ('(', '_.__.'),
    (')', '_.__._'), (',', '__..__'), ('-', '_...._'), ('.', '._._._'),
    ('/', '_.._.'), (':', '___...'), (';', '_._._.'), ('?', '..__..'),
    ('_', '..__._'), ('@', '.__._.'), ('!', '_._.__')
]


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
