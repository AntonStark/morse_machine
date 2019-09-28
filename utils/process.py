import numpy as np
import pandas as pd
import wave
from matplotlib import mlab
from scipy.signal import butter, sosfiltfilt

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


#
# WAV FILE STUFF
#

def load_raw(filepath, seconds=None):
    wave_file = wave.open(filepath, 'rb')
    nframes, framerate, sampwidth = wave_file.getnframes(), wave_file.getframerate(), wave_file.getsampwidth()
    if seconds:
        start, end = (s * sampwidth * framerate for s in seconds)  # "2" because of frame size
        if end > sampwidth * nframes:
            raise IndexError(
                f'file {filepath} duration: {nframes / framerate}s., interval=({seconds[0]}, {seconds[1]}) given')
        wav_frames = wave_file.readframes(nframes)[int(start):int(end)]
    else:
        wav_frames = wave_file.readframes(nframes)
    wave_file.close()
    if sampwidth == 2:
        ys = np.frombuffer(wav_frames, dtype=np.int16)
    elif sampwidth == 4:
        ys = np.frombuffer(wav_frames, dtype=np.int32)
    else:
        raise ValueError(f'unexpected wav sampwidth={sampwidth}')
    return ys, wave_file


def time_labels(wave_file: wave.Wave_read, points=None):
    if points is None:
        points = wave_file.getnframes()
    ts = np.linspace(0, wave_file.getnframes() / wave_file.getframerate(), num=points)
    return ts


def time_labels_interval(wf: wave.Wave_read, seconds, points=None):
    if seconds:
        labels = np.arange(seconds[0], seconds[1], 1. / wf.getframerate())
        if points:
            start = int((len(labels) - points) / 2)
            end = start + points
            return labels[start:end]
        else:
            return labels
    else:
        if points is None:
            points = wf.getnframes()
        labels = np.linspace(0, wf.getnframes() / wf.getframerate(), num=points)
        return labels


#
# SPECTRUM STUFF
#

def calc_spectrum(data, fs):
    nfft, noverlap = 512, 384

    spectrum, freqs, t = mlab.specgram(data, Fs=fs, NFFT=nfft,
                                       noverlap=noverlap, detrend='none')
    pad_xextent = (nfft - noverlap) / fs / 2
    xmin, xmax = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    spec_extent = xmin, xmax, freqs[0], freqs[-1]
    return spectrum, spec_extent


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = butter(order, [low, high], btype='band')
    # return b, a
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(input_signal, lowcut, highcut, fs, order=5):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # output = lfilter(b, a, input)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    output = sosfiltfilt(sos, input_signal)
    return output


def adaptive_bandpass_filter(input_signal, spectrum, fs, gap=25, order=3):
    freq_en_dist = spectrum.sum(axis=1)
    main_freq = freq_en_dist.argmax() / len(freq_en_dist) * fs / 2
    sos = butter_bandpass(main_freq - gap, main_freq + gap, fs, order=order)
    output = sosfiltfilt(sos, input_signal)
    return output


#
# DOT AND DASH DURATIONS
#

def split_sign_series(ds):
    beats = []
    nulls = []

    def push(sign, idx_list):
        if sign:
            beats.append(idx_list)
        else:
            nulls.append(idx_list)

    buf = []
    pr = ds[0]
    for idx, val in enumerate(ds):
        if val == pr:
            buf.append(idx)
        else:
            push(pr, buf)
            buf = [idx]

        pr = val
    push(pr, buf)
    return nulls, beats


def intervals_data(amplitude, intervals, inter_agg=np.mean):
    df = pd.DataFrame(index=np.arange(0, len(intervals)),
                      columns=('p_mid', 'dur', 'ampl_extr'))
    for r, idx_list in enumerate(intervals):
        df.loc[r] = [
            np.median(idx_list), len(idx_list),
            inter_agg(amplitude[idx_list])
        ]
    return df


def sign_series(ds):
    data = []

    def push(idx_list):
        label = 'beat' if ds[idx_list[0]] else 'null'
        count = len(idx_list)
        mid = idx_list[int((count - 1) / 2)]
        data.append([mid, label, count, idx_list])

    buf = []
    pr = ds[0]
    for idx, val in enumerate(ds):
        if val == pr:
            buf.append(idx)
        else:
            push(buf)
            buf = [idx]
        pr = val
    push(buf)

    df = pd.DataFrame(data, columns=('idx_mid', 'label', 'count', 'indices'))
    return df


#
# DECODE
#

def morse2text(morse):
    """
    :param morse: list of str which is either ' ' or consists of '.' and '_"
    :return: decoded string
    """
    mapping = {morse: ch for ch, morse in CODE}
    decoded = [mapping[m] if m in mapping else '#'
               for m in morse]
    return ''.join(decoded)


def predict(idf, dot_len, factor=0.3):
    """
    :param idf: pd.DataFrame(data, columns=('idx_mid', 'label', 'count', 'indices'))
    :param dot_len: len in points
    :param factor:
    :return:
    """
    def classify(n_points, label):
        dot_len_min = (1 - factor) * dot_len
        dot_len_max = (1 + factor) * dot_len
        if label == 'beat':
            if dot_len_min <= n_points <= dot_len_max:
                return '.'
            elif 3 * dot_len_min <= n_points <= 3 * dot_len_max:
                return '_'
            else:
                return None
        elif label == 'null':
            if dot_len_min <= n_points <= dot_len_max:
                return ''
            elif 3 * dot_len_min <= n_points <= 3 * dot_len_max:
                return '|'
            elif 7 * dot_len_min <= n_points:
                return ' '          # silence during 7 dots AND MORE
            else:
                return None
        else:
            return None
    raw_result = [classify(count, label) for _, label, count in idf[['label', 'count']].itertuples()]
    result = [r for r in raw_result if r is not None and r != '']
    morse_seq = ''.join(result).replace(' ', '| |').split('|')
    result = morse2text(morse_seq).upper()
    return result
