import os
import numpy as np
import pandas as pd
from matplotlib import mlab
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression

import utils
from encode import CODE

AUDIO_DIR = './audio'
FS = 8000
FILENAME = 'cw001.wav'
SECONDS = (0, 6)
WINDOWS = True


def time_labels(wave_file, points=None):
    if points is None:
        points = wave_file.getnframes()
    ts = np.linspace(0, wave_file.getnframes() / wave_file.getframerate(), num=points)
    return ts


def time_labels_interval(wf, points=None):
    labels = np.arange(SECONDS[0], SECONDS[1], 1. / wf.getframerate())
    if points:
        start = int((len(labels) - points) / 2)
        end = start + points
        return labels[start:end]
    else:
        return labels



def calc_spectrum(data):
    nfft, noverlap, fs = 512, 384, FS

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


def adaptive_bandpass_filter(input, spectrum, gap=25, order=3):
    freq_en_dist = spectrum.sum(axis=1)
    main_freq = freq_en_dist.argmax() / len(freq_en_dist) * FS / 2
    sos = butter_bandpass(main_freq - gap, main_freq + gap, FS, order=order)
    output = sosfiltfilt(sos, input)
    return output


def draw_time_series(*ys, filename, wave_file):
    # ts = time_labels(wave_file, len(ys))
    for y in ys:
        plt.plot(time_labels_interval(wave_file, len(y)), y)
    plt.gcf().set_size_inches(15, 5)
    plt.gca().set_xlim(*SECONDS)
    if WINDOWS:
        plt.show()
    else:
        plt.savefig(filename, dpi=100)
        plt.close()


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


def intervals_data(amplitude, intervals, inter_agg=np.mean):
    df = pd.DataFrame(index=np.arange(0, len(intervals)),
                      columns=('p_mid', 'dur', 'ampl_extr'))
    for r, idx_list in enumerate(intervals):
        df.loc[r] = [
            np.median(idx_list), len(idx_list),
            inter_agg(amplitude[idx_list])
        ]
    return df


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
    def classify(len, label, dot_len, factor):
        dot_len_min = (1 - factor) * dot_len
        dot_len_max = (1 + factor) * dot_len
        if label == 'beat':
            if dot_len_min <= len <= dot_len_max:
                return '.'
            elif 3 * dot_len_min <= len <= 3 * dot_len_max:
                return '_'
            else:
                return None
        elif label == 'null':
            if dot_len_min <= len <= dot_len_max:
                return ''
            elif 3 * dot_len_min <= len <= 3 * dot_len_max:
                return '|'
            elif 7 * dot_len_min <= len:
                return ' '          # silence during 7 dots AND MORE
            else:
                return None
        else:
            return None
    raw_result = [classify(count, label, dot_len, factor) for _, label, count in idf[['label', 'count']].itertuples()]
    result = [r for r in raw_result if r is not None and r != '']
    morse_seq = ''.join(result).replace(' ', '| |').split('|')
    result = morse2text(morse_seq).upper()
    return result


def main():
    filepath = os.path.join(AUDIO_DIR, FILENAME)
    raw_values, wave_file = utils.process.load_raw(filepath, SECONDS)
    spectrum, spec_extent = calc_spectrum(raw_values)
    filtered_values = adaptive_bandpass_filter(raw_values, spectrum)
    amplitude = pd.Series(filtered_values).apply(np.abs)
    ampl_sm: pd.Series = amplitude.rolling(72, center=True).mean()
    draw_time_series(ampl_sm, filename='amplitude.png', wave_file=wave_file)

    default_dash_duration = 1000
    window_size = 6 * default_dash_duration

    window_min = ampl_sm.rolling(window_size, center=True, min_periods=int(0.1 * window_size)).min()
    window_max = ampl_sm.rolling(window_size, center=True, min_periods=int(0.1 * window_size)).max()
    min_max_threshold: pd.Series = (window_max + window_min) / 2
    draw_time_series(ampl_sm, min_max_threshold, filename='amplitude_minmax.png', wave_file=wave_file)

    discr_step_one = ampl_sm > min_max_threshold
    # noinspection PyTypeChecker
    nulls, beats = split_sign_series(discr_step_one)
    # noinspection PyTypeChecker
    nnb_inter = sign_series(discr_step_one)
    null_info = intervals_data(ampl_sm, nulls, np.min)
    beat_info = intervals_data(ampl_sm, beats, np.max)

    tl = time_labels_interval(wave_file, len(ampl_sm))
    plt.plot(tl, ampl_sm, tl, min_max_threshold)
    plt.plot(SECONDS[0] + (null_info['p_mid'] / FS), null_info['ampl_extr'], 'ko')
    plt.plot(SECONDS[0] + (beat_info['p_mid'] / FS), beat_info['ampl_extr'], 'r*')
    plt.gcf().set_size_inches(15, 5)
    plt.gca().set_xlim(*SECONDS)
    if WINDOWS:
        plt.show()
    else:
        plt.savefig('amplitude_minmax_points.png', dpi=100)
        plt.close()

    plt.hist(beat_info['dur'] / FS, bins=100)
    plt.gca().set_title('dash&dot')
    if WINDOWS:
        plt.show()
    else:
        plt.savefig('beats_hist.png', dpi=100)
        plt.close()

    plt.hist(null_info['dur'] / FS, bins=100)
    plt.gca().set_title('null')
    if WINDOWS:
        plt.show()
    else:
        plt.savefig('nulls_hist.png', dpi=100)
        plt.close()

    peak_filtered = beat_info[beat_info['dur'] / FS > 0.01]
    peak_points_x, peak_points_y = peak_filtered['p_mid'], peak_filtered['ampl_extr']
    peak_tck = interpolate.splrep(peak_points_x, peak_points_y)

    def interp(n_point):
        return interpolate.splev(n_point, peak_tck)
    ip = np.vectorize(interp)
    max_inter = ip(np.arange(0, len(tl)))

    low_filtered = null_info[null_info['dur'] / FS > 0.1]
    low_points_x, low_points_y = low_filtered['p_mid'], low_filtered['ampl_extr']
    low_tck = interpolate.splrep(low_points_x, low_points_y)

    def interp(n_point):
        return interpolate.splev(n_point, low_tck)

    il = np.vectorize(interp)
    min_inter = il(np.arange(0, len(tl)))

    plt.plot(tl, ampl_sm)
    plt.plot(tl, max_inter, 'r')
    plt.plot(SECONDS[0] + (peak_points_x / FS), peak_points_y, 'rx')
    plt.plot(tl, min_inter, 'k')
    plt.plot(SECONDS[0] + (low_points_x / FS), low_points_y, 'kx')
    plt.plot(tl, (min_inter + max_inter) / 2, 'm')
    plt.gcf().set_size_inches(15, 5)
    plt.gca().set_xlim(*SECONDS)
    if WINDOWS:
        plt.show()
    else:
        plt.savefig('full_step_one.png', dpi=100)
        plt.close()

    di = beat_info['dur']
    idi = di[di > 100]
    # beat_len_delta, desired_bin_width = idi.max() - idi.min(), 30
    # np.histogram(idi, bins=(int(beat_len_delta / desired_bin_width)))
    dots, dashes = idi[idi < idi.mean()], idi[idi > idi.mean()]
    plt.plot(len(dots) * [1], dots)
    plt.plot(len(dashes) * [3], dashes)
    plt.gca().set_xlim(0, 3.5)
    plt.gca().set_ylim(0, 3000)
    plt.show()
    reg = LinearRegression().fit(
        np.array(len(dots) * [1] + len(dashes) * [3]).reshape(-1, 1),
        np.concatenate([np.array(dots).astype(np.int), np.array(dashes).astype(np.int)])
    )
    dot_len = reg.predict([[1.]])[0]
    # print(f'dot={dot_len}')

    result = predict(nnb_inter, dot_len)
    print(result)


if __name__ == '__main__':
    if WINDOWS:
        plt.interactive(True)
    main()
