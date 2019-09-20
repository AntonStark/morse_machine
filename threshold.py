import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.linear_model import LinearRegression

import utils
from encode import CODE

AUDIO_DIR = './audio'
PLOTS_DIR = './plots'
CSV_DIR = './csv_tables'

FILENAME = 'cw001.wav'

FS = 8000
SECONDS = (0, 6)
INTERACTIVE = False


def main():
    # LOAD, FILTER, GET SMOOTH
    raw_values, wave_file = utils.process.load_raw(utils.audio_filepath(AUDIO_DIR, FILENAME), SECONDS)
    spectrum, spec_extent = utils.process.calc_spectrum(raw_values, FS)
    filtered_values = utils.process.adaptive_bandpass_filter(raw_values, spectrum, FS)
    amplitude = pd.Series(filtered_values).apply(np.abs)
    ampl_sm: pd.Series = amplitude.rolling(72, center=True).mean()
    utils.plot.draw_time_series(ampl_sm, wave_file=wave_file, seconds=SECONDS,
                                interactive=INTERACTIVE, save_to=utils.plot_filepath(PLOTS_DIR, 'amplitude.png'))

    # FIND THRESHOLD FUNCTION
    default_dash_duration = 1000    # in points
    window_size = 6 * default_dash_duration

    window_min = ampl_sm.rolling(window_size, center=True, min_periods=int(0.1 * window_size)).min()
    window_max = ampl_sm.rolling(window_size, center=True, min_periods=int(0.1 * window_size)).max()
    min_max_threshold: pd.Series = (window_max + window_min) / 2
    utils.plot.draw_time_series(ampl_sm, min_max_threshold, wave_file=wave_file, seconds=SECONDS,
                                interactive=INTERACTIVE, save_to=utils.plot_filepath(PLOTS_DIR, 'amplitude_minmax.png'))

    # BEATS DURATION
    discr_step_one = ampl_sm > min_max_threshold
    # noinspection PyTypeChecker
    nulls, beats = utils.process.split_sign_series(discr_step_one)
    null_info = utils.process.intervals_data(ampl_sm, nulls, np.min)
    beat_info = utils.process.intervals_data(ampl_sm, beats, np.max)

    utils.plot.draw_time_series_with_points(ampl_sm, min_max_threshold, beat_info=beat_info, null_info=null_info,
                                            wave_file=wave_file, fs=FS, seconds=SECONDS,
                                            interactive=INTERACTIVE,
                                            save_to=utils.plot_filepath(PLOTS_DIR, 'amplitude_minmax_points.png'))

    utils.plot.draw_hist(beat_info['dur'], FS, 'dash&dot',
                         INTERACTIVE, save_to=utils.plot_filepath(PLOTS_DIR, 'beats_hist.png'))
    utils.plot.draw_hist(null_info['dur'], FS, 'null',
                         INTERACTIVE, save_to=utils.plot_filepath(PLOTS_DIR, 'nulls_hist.png'))

    peak_filtered = beat_info[beat_info['dur'] / FS > 0.01]
    peak_points_x, peak_points_y = peak_filtered['p_mid'], peak_filtered['ampl_extr']
    peak_tck = interpolate.splrep(peak_points_x, peak_points_y)

    tl = utils.process.time_labels_interval(wave_file, SECONDS, len(ampl_sm))

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
    if INTERACTIVE:
        plt.show()
    else:
        fn = utils.plot_filepath(PLOTS_DIR, 'full_step_one.png')
        plt.savefig(fn, dpi=100)
        plt.close()

    di = beat_info['dur']
    idi = di[di > 100]
    # beat_len_delta, desired_bin_width = idi.max() - idi.min(), 30
    # np.histogram(idi, bins=(int(beat_len_delta / desired_bin_width)))
    dots, dashes = idi[idi < idi.mean()], idi[idi > idi.mean()]

    reg = LinearRegression().fit(
        np.array(len(dots) * [1] + len(dashes) * [3]).reshape(-1, 1),
        np.concatenate([np.array(dots).astype(np.int), np.array(dashes).astype(np.int)])
    )
    r1_dur, dot_len, r2_dur = reg.predict([[0.], [1.], [3.5]])

    plt.plot(len(dots) * [1], dots)
    plt.plot(len(dashes) * [3], dashes)
    plt.plot([0, 3.5], [r1_dur, r2_dur])
    plt.gca().set_xlim(0, 3.5)
    plt.gca().set_ylim(0, 3000)
    if INTERACTIVE:
        plt.show()
    else:
        fn = utils.plot_filepath(PLOTS_DIR, 'durations_regr.png')
        plt.savefig(fn, dpi=100)
        plt.close()

    # noinspection PyTypeChecker
    nnb_inter = utils.process.sign_series(discr_step_one)
    result = utils.process.predict(nnb_inter, dot_len)
    print(result)


if __name__ == '__main__':
    if INTERACTIVE:
        plt.interactive(True)
    main()
