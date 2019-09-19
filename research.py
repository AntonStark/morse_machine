import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

import utils

AUDIO_DIR = './audio'
PLOTS_DIR = './plots'

TEST1 = 'cw001.wav'

FS = 8000


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


def plot_spectrum(spectrum, extent, filename):
    z = 10. * np.log10(spectrum)
    z = np.flipud(z)
    plt.imshow(z, plt.magma(), extent=extent, aspect='auto')
    plt.gcf().set_size_inches(15, 5)
    plt.savefig(filename, dpi=100)
    plt.close()


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
    ys, wave_file = utils.process.load_raw(utils.audio_filepath(AUDIO_DIR, TEST1), seconds=(0, 6))

    utils.plot.draw_time_series(ys, wave_file=wave_file, seconds=(0, 6), interactive=False, save_to='amplitude.png')

    # draw_ampl_dist(ys, 'dist.png')
    draw_ampl_dist(ys, 'dist.png', bins=100)
    draw_ampl_log_dist(ys, 'log_dist.png', bins=100)

    spectrum, spec_extent  = utils.process.calc_spectrum(ys, FS)
    plot_spectrum(spectrum, spec_extent, 'raw_spectrum.png')

    fys = utils.process.adaptive_bandpass_filter(ys, spectrum, FS)

    utils.plot.draw_time_series(fys, wave_file=wave_file, seconds=(0, 6), interactive=False,
                                save_to='filtered_amplitude.png')
    draw_ampl_dist(fys, 'dist_filtered.png', bins=100)

    spectrum_f, spec_f_extent = utils.process.calc_spectrum(fys, FS)
    plot_spectrum(spectrum_f, spec_f_extent, 'filtered_spectrum.png')

    sm0 = pd.Series(fys).apply(np.abs)

    def draw_smooth(period, source=sm0, filename = None):
        if not filename:
            filename = f'sm{period}_amplitude.png'
        sm = source.rolling(period).mean()
        plt.plot(utils.process.time_labels(wave_file, len(sm)), sm)
        plt.gcf().set_size_inches(15, 5)
        plt.gca().set_ylim(0, 2000)
        plt.gca().set_xlim(0, 6)
        plt.savefig(filename, dpi=100)
        plt.close()

    # draw_smooth(0)
    # draw_smooth(3)      # 0.4 ms
    # draw_smooth(16)     # 2 ms
    # draw_smooth(48)     # 6 ms
    draw_smooth(72)     # 9 ms
    # draw_smooth(200)    # 25 ms
    # draw_smooth(400)    # 50 ms

    # fys20 = adaptive_bandpass_filter(ys, spectrum, gap=20)
    # sm20 = pd.Series(fys20).apply(np.abs)
    # draw_smooth(72, sm20, 'sm72_fys20.png')  # 9 ms
    # draw_smooth(48, sm20, 'sm48_fys20.png')  # 9 ms
    #
    # fys15 = adaptive_bandpass_filter(ys, spectrum, gap=15)
    # sm15 = pd.Series(fys15).apply(np.abs)
    # draw_smooth(72, sm15, 'sm72_fys15.png')     # 9 ms
    # draw_smooth(48, sm15, 'sm48_fys15.png')     # 9 ms
    #
    # fys10 = adaptive_bandpass_filter(ys, spectrum, gap=10)
    # sm10 = pd.Series(fys10).apply(np.abs)
    # draw_smooth(72, sm10, 'sm72_fys10.png')  # 9 ms
    # draw_smooth(48, sm10, 'sm48_fys10.png')  # 9 ms

    sm = sm0.rolling(72).mean()
    log = np.log(sm)
    plt.plot(utils.process.time_labels(wave_file, len(log)), log)
    plt.gcf().set_size_inches(15, 5)
    # plt.gca().set_ylim(0, 2000)
    plt.gca().set_xlim(0, 6)
    plt.savefig('log.png', dpi=100)
    plt.close()

    def draw_peaks(distance):
        sm_not_nan = sm
        sm_not_nan[np.isnan(sm)] = 0.
        peaks, _ = find_peaks(sm_not_nan, distance=distance)
        tl = utils.process.time_labels(wave_file, len(sm_not_nan))
        plt.plot(tl, sm_not_nan)
        plt.plot(tl[peaks], sm_not_nan[peaks], 'x')
        plt.gcf().set_size_inches(15, 5)
        plt.gca().set_ylim(0, 2000)
        plt.gca().set_xlim(0, 6)
        plt.savefig(f'peaks{distance}ms.png', dpi=100)
        plt.close()
    draw_peaks(160)
    draw_peaks(320)
    draw_peaks(800)

    def draw_low_high_medeians(period):
        def lower_half_median(sample):
            sample = sample[~np.isnan(sample)]
            lower_values = np.sort(sample)[:int(len(sample) / 2)]
            return np.median(lower_values)

        def upper_half_median(sample):
            sample = sample[~np.isnan(sample)]
            upper_values = np.sort(sample)[int(len(sample) / 2):]
            return np.median(upper_values)

        # low_min = sm.rolling(period, center=True, min_periods=400).min()
        low_med = sm.rolling(period, center=True, min_periods=400).apply(lower_half_median, raw=True)
        high_med = sm.rolling(period, center=True, min_periods=400).apply(upper_half_median, raw=True)
        # high_med1 = sm.rolling(int(0.8 * period), center=True, min_periods=400).apply(upper_half_median)
        # high_med2 = sm.rolling(int(0.5 * period), center=True, min_periods=400).apply(upper_half_median)
        # high_med3 = sm.rolling(int(1.2 * period), center=True, min_periods=400).apply(upper_half_median)
        # high_med4 = sm.rolling(int(1.5 * period), center=True, min_periods=400).apply(upper_half_median)
        tl = utils.process.time_labels(wave_file, len(sm))
        plt.plot(tl, sm)
        plt.plot(tl, high_med, 'g')
        # plt.plot(tl, high_med, label=f'{period}')
        # plt.plot(tl, high_med1, label=f'{int(0.8 * period)}')
        # plt.plot(tl, high_med2, label=f'{int(0.5 * period)}')
        # plt.plot(tl, high_med3, label=f'{int(1.2 * period)}')
        # plt.plot(tl, high_med4, label=f'{int(1.5 * period)}')
        # plt.plot(tl, low_min, 'm')
        plt.plot(tl, low_med, 'r')

        plt.gcf().set_size_inches(30, 5)
        # plt.gca().legend()
        plt.gca().set_ylim(0, 2000)
        plt.gca().set_xlim(0, 12)
        plt.savefig(f'median{period}full_step_one.png', dpi=100)
        plt.close()

        norm = (sm - low_med) / high_med
        plt.plot(tl, sm)
        plt.plot(tl, 1000 * norm)
        plt.gcf().set_size_inches(30, 5)
        plt.gca().set_ylim(0, 2000)
        plt.gca().set_xlim(0, 12)
        plt.savefig(f'normalized{period}full_step_one.png', dpi=100)
        plt.close()
        return norm

    n1 = draw_low_high_medeians(8000)
    # draw_low_high_medeians(16000)
    # draw_low_high_medeians(32000)

    def draw_max_min(period):
        low_ = sm.rolling(period, center=True, min_periods=int(period / 10)).min()
        high = sm.rolling(period, center=True, min_periods=int(period / 10)).max()
        tl = utils.process.time_labels(wave_file, len(sm))
        plt.plot(tl, sm)
        plt.plot(tl, low_, 'b:')
        plt.plot(tl, high, 'r:')
        plt.plot(tl, (low_ + high) / 2., 'g:')

        plt.gcf().set_size_inches(30, 5)
        # plt.gca().legend()
        plt.gca().set_ylim(0, 2000)
        plt.gca().set_xlim(0, 12)
        plt.savefig(f'max_min_{period}full_step_one.png', dpi=100)
        plt.close()

    draw_max_min(4000)
    draw_max_min(6000)
    draw_max_min(8000)

    # dd1 = threshold(sm, 160)
    dd1 = threshold(n1, 0.5)
    # todo число связных компонент "1" в зависимости от порогового значения
    #  поскольку импульсы имеют некоторую высоту, график числа компонент от величины
    #  порогового значения будет иметь "полочку"
    # todo также можно смотреть на распределнеие длительностей "0" при разных пороговых
    plt.plot(utils.process.time_labels(wave_file, len(dd1)), dd1, fillstyle='none')
    plt.gca().set_xlim(0, 6)
    plt.gcf().set_size_inches(15, 5)
    plt.savefig('discrete.png', dpi=100)
    plt.close()

    null_len, beat_len = series_length(dd1)
    print(f'nulls: count={len(null_len)}, min_len={np.min(null_len)}, max_len={np.max(null_len)}')
    print(f'beats: count={len(beat_len)}, min_len={np.min(beat_len)}, max_len={np.max(beat_len)}')

    utils.plot.draw_hist(beat_len, FS, 'dash&dot',
                         False, save_to=utils.plot_filepath(PLOTS_DIR, 'beats_hist.png'))
    utils.plot.draw_hist(null_len, FS, 'null',
                         False, save_to=utils.plot_filepath(PLOTS_DIR, 'nulls_hist.png'))
    debug = []


if __name__ == '__main__':
    main()
