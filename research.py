import os
import numpy as np
import pandas as pd
from matplotlib import mlab as mlab
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfiltfilt
from wave import open as open_wave

AUDIO_DIR = './audio'
FS = 8000
TEST1 = 'cw001.wav'


def load_raw(name):
    filepath = os.path.join(AUDIO_DIR, name)
    wave_file = open_wave(filepath, 'rb')
    nframes = wave_file.getnframes()
    wav_frames = wave_file.readframes(nframes)
    ys = np.frombuffer(wav_frames, dtype=np.int16)
    return ys, wave_file


def time_labels(wave_file, len=None):
    if len is None:
        len = wave_file.getnframes()
    ts = np.linspace(0, wave_file.getnframes() / wave_file.getframerate(), num=len)
    return ts


def draw_amplitude(ts, ys, filename):
    plt.plot(ts, ys)
    plt.gcf().set_size_inches(15, 5)
    plt.gca().set_ylim(-3500, 3500)
    plt.gca().set_xlim(0, 6)
    plt.savefig(filename, dpi=100)
    plt.close()


def draw_ampl_dist(ys, name, **kwargs):
    ms = pd.DataFrame(ys).apply(abs)
    ms.hist(**kwargs)
    plt.savefig(name, dpi=100)
    plt.close()


def draw_ampl_log_dist(ys, name, **kwargs):
    ms = pd.DataFrame(ys).apply(abs).apply(np.log)
    tr = ms[ms[0] > 0]
    tr.hist(**kwargs)
    plt.savefig(name, dpi=100)
    plt.close()


def calc_spectrum(data):
    nfft, noverlap, fs = 512, 384, FS

    spectrum, freqs, t = mlab.specgram(data, Fs=fs, NFFT=nfft,
                                       noverlap=noverlap, detrend='none')
    pad_xextent = (nfft - noverlap) / fs / 2
    xmin, xmax = np.min(t) - pad_xextent, np.max(t) + pad_xextent
    spec_extent = xmin, xmax, freqs[0], freqs[-1]
    return spectrum, spec_extent


def plot_spectrum(spectrum, extent, filename):
    z = 10. * np.log10(spectrum)
    z = np.flipud(z)
    plt.imshow(z, plt.magma(), extent=extent, aspect='auto')
    plt.gcf().set_size_inches(15, 5)
    plt.savefig(filename, dpi=100)
    plt.close()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = butter(order, [low, high], btype='band')
    # return b, a
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(input, lowcut, highcut, fs, order=5):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # output = lfilter(b, a, input)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    output = sosfiltfilt(sos, input)
    return output


def adaptive_bandpass_filter(input, spectrum, gap=25, order=3):
    freq_en_dist = spectrum.sum(axis=1)
    main_freq = freq_en_dist.argmax() / len(freq_en_dist) * FS / 2
    sos = butter_bandpass(main_freq - gap, main_freq + gap, FS, order=order)
    output = sosfiltfilt(sos, input)
    return output


def threshold(data, value):
    return np.abs(data) > value


def series_length(ds):
    beat_len = []
    null_len = []
    def push(s, l):
        if s:
            beat_len.append(l)
        else:
            null_len.append(l)

    pr, l = None, 0
    for v in ds:
        if pr is None:
            pr = v
        if pr == v:
            l += 1
        else:
            push(pr, l)
            l = 1
        pr = v
    push(pr, l)
    return null_len, beat_len


def main():
    ys, wave_file = load_raw(TEST1)
    # print(ys, len(ys))
    # wave_file.close()

    ts = time_labels(wave_file)
    draw_amplitude(ts, ys, 'amplitude.png')

    # draw_ampl_dist(ys, 'dist.png')
    draw_ampl_dist(ys, 'dist.png', bins=100)
    draw_ampl_log_dist(ys, 'log_dist.png', bins=100)

    spectrum, spec_extent  = calc_spectrum(ys)
    plot_spectrum(spectrum, spec_extent, 'raw_spectrum.png')

    fys = adaptive_bandpass_filter(ys, spectrum)
    fts = time_labels(wave_file, len(fys))
    draw_amplitude(fts, fys, 'filtered_amplitude.png')
    draw_ampl_dist(fys, 'dist_filtered.png', bins=100)

    spectrum_f, spec_f_extent = calc_spectrum(fys)
    plot_spectrum(spectrum_f, spec_f_extent, 'filtered_spectrum.png')

    loudness = pd.Series(fys).rolling(8, min_periods=1).max()
    ampl_abs = pd.Series(fys).apply(np.abs)

    def draw_smooth(period):
        sm1 = ampl_abs.rolling(period, center=True).mean()
        tl = time_labels(wave_file, len(sm1))
        if period > 0:
            plt.plot(tl, sm1)
        else:
            plt.plot(tl, ampl_abs)
        plt.gcf().set_size_inches(15, 5)
        plt.gca().set_ylim(0, 2000)
        plt.gca().set_xlim(0, 6)
        plt.savefig(f'sm{period}_amplitude.png', dpi=100)
        plt.close()

    draw_smooth(0)
    draw_smooth(16)     # 2 ms
    draw_smooth(48)     # 6 ms
    draw_smooth(72)     # 9 ms
    draw_smooth(200)    # 25 ms

    def smooth72_loudness(period):
        loudness = ampl_abs.rolling(period, min_periods=1, center=True, win_type='hamming').mean()

        sm = ampl_abs.rolling(72, center=True).mean()
        # sm_loud = loudness.rolling(72, center=True).mean()
        norm = sm / loudness * 1000.

        tl = time_labels(wave_file, len(sm))

        plt.plot(tl, sm, 'b-', tl, loudness, 'r:', tl, norm, 'g:')
        plt.gcf().set_size_inches(15, 5)
        plt.gca().set_ylim(0, 3000)
        plt.gca().set_xlim(0, 6)
        plt.savefig(f'sm_loud{period}.png', dpi=100)
        plt.close()

    # smooth72_loudness(8)        # 1 ms
    # smooth72_loudness(400)      # 50 ms
    # smooth72_loudness(800)      # 100 ms
    smooth72_loudness(4000)     # 500 ms
    smooth72_loudness(8000)     # 1000 ms

    # draw_smooth(400)    # 50 ms
    sm = ampl_abs.rolling(72, center=True).mean()
    # print(f'len(sm)={len(sm)}')
    #
    # temp = sm - sm.min()
    # sm_ft = temp.apply(lambda v: v if v > temp.max() / 20. else np.NaN)
    #
    # loud05 = sm_ft.rolling(6000, center=True, win_type='triang', min_periods=3000).mean()
    # loud1 = sm_ft.rolling(8000, center=True, win_type='triang', min_periods=4000).mean()
    # loud2 = sm_ft.rolling(12000, center=True, win_type='triang', min_periods=6000).mean()
    #
    # norm = sm_ft / loud2 * 1000
    # print(f'len(loud)={len(loud05)}')
    #
    # tl = time_labels(wave_file, len(sm))
    # plt.plot(tl, sm_ft, '-', tl, loud05, 'r:', tl, loud1, 'g:', tl, loud2, 'b:', tl, norm, 'm:')
    # plt.gcf().set_size_inches(15, 5)
    # plt.gca().set_xlim(0, 6)
    # plt.savefig(f'sm_amp72__loud4000.png', dpi=100)
    # plt.close()

    dd1 = threshold(sm, 160)
    # todo число связных компонент "1" в зависимости от порогового значения
    #  поскольку импульсы имеют некоторую высоту, график числа компонент от величины
    #  порогового значения будет иметь "полочку"
    # todo также можно смотреть на распределнеие длительностей "0" при разных пороговых
    plt.plot(time_labels(wave_file, len(dd1)), dd1, fillstyle='none')
    plt.gca().set_xlim(0, 6)
    plt.gcf().set_size_inches(15, 5)
    plt.savefig('discrete.png', dpi=100)
    plt.close()

    null_len, beat_len = series_length(dd1)
    print(f'nulls: count={len(null_len)}, min_len={np.min(null_len)}, max_len={np.max(null_len)}')
    print(f'beats: count={len(beat_len)}, min_len={np.min(beat_len)}, max_len={np.max(beat_len)}')

    ndf = pd.DataFrame(null_len)
    ndf[ndf[0] > 1].hist(bins=100)
    # plt.gca().set_ylim(0, 60)
    plt.savefig('null.png', dpi=100)
    plt.close()

    bdf = pd.DataFrame(beat_len)
    bdf[bdf[0] > 1].hist(bins=100)
    # plt.gca().set_ylim(0, 10)
    plt.savefig('beat.png', dpi=100)
    plt.close()
    # plt.subplot(2, 1, 1)
    # hist, bins = np.histogram(null_len, bins=100)
    # plt.plot(bins[:-1], hist)
    # plt.subplot(2, 1, 2)
    # hist, bins = np.histogram(beat_len, bins=100)
    # plt.plot(bins[:-1], hist)
    # plt.gcf().savefig('null_beat.png')
    debug = []


if __name__ == '__main__':
    main()
