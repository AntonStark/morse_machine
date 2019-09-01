import os
import numpy as np
import pandas as pd
from matplotlib import mlab
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfiltfilt
from wave import open as open_wave

AUDIO_DIR = './audio'
FS = 8000
FILENAME = 'cw050.wav'
SECONDS = (6, 11)
WINDOWS = True


def load_raw(name):
    filepath = os.path.join(AUDIO_DIR, name)
    wave_file = open_wave(filepath, 'rb')
    nframes, framerate = wave_file.getnframes(), wave_file.getframerate()
    start, end = (2 * s * framerate for s in SECONDS)       # "2" because of frame size
    if end > 2 * nframes:
        raise IndexError(f'file {FILENAME} duration: {nframes / framerate}s., interval=({SECONDS[0]}, {SECONDS[1]}) given')
    wav_frames = wave_file.readframes(nframes)[int(start):int(end)]
    ys = np.frombuffer(wav_frames, dtype=np.int16)
    return ys, wave_file


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


def intervals_data(amplitude, intervals):
    df = pd.DataFrame(index=np.arange(0, len(intervals)),
                      columns=('p_mid', 'dur', 'ampl_mean'))
    for r, idx_list in enumerate(intervals):
        df.loc[r] = [np.median(idx_list), len(idx_list), np.mean(amplitude[idx_list])]
    return df


def main():
    raw_values, wave_file = load_raw(FILENAME)
    spectrum, spec_extent  = calc_spectrum(raw_values)
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
    nulls, beats = split_sign_series(discr_step_one)
    null_info = intervals_data(ampl_sm, nulls)
    beat_info = intervals_data(ampl_sm, beats)

    tl = time_labels_interval(wave_file, len(ampl_sm))
    plt.plot(tl, ampl_sm, tl, min_max_threshold)
    plt.plot(SECONDS[0] + (null_info['p_mid'] / FS), null_info['ampl_mean'], 'ko')
    plt.plot(SECONDS[0] + (beat_info['p_mid'] / FS), beat_info['ampl_mean'], 'r*')
    plt.gcf().set_size_inches(15, 5)
    plt.gca().set_xlim(*SECONDS)
    if WINDOWS:
        plt.show()
    else:
        plt.savefig('amplitude_minmax_points.png', dpi=100)
        plt.close()

    plt.hist(beat_info['dur'], bins=100)
    plt.show()


if __name__ == '__main__':
    if WINDOWS:
        plt.interactive(True)
    main()
