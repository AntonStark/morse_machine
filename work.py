import os

import utils

AUDIO_DIR = './audio'


def make_predictions(file_numbers):
    for num in file_numbers:
        filepath = os.path.join(AUDIO_DIR, 'cw{:0>3}.wav'.format(num))
        ys, wave_file = utils.process.load_raw(filepath)


def main():
    pass


if __name__ == '__main__':
    main()
