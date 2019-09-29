import os
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.linear_model import LinearRegression

import utils

AUDIO_DIR = './audio'
CSV_DIR = './csv_tables'

LABELS_CSV = os.path.join(CSV_DIR, 'MorseLabels.csv')
PREDICTIONS_CSV = os.path.join(CSV_DIR, 'Predictions2.csv')


def make_predictions(file_numbers):
    # file_numbers = [52]      # debug only
    predictions = {}
    for i, num in enumerate(file_numbers):
        # load
        name = 'cw{:0>3}.wav'.format(num)
        filepath = os.path.join(AUDIO_DIR, name)
        raw_values, wave_file = utils.process.load_raw(filepath)
        fs = wave_file.getframerate()

        # filter noise
        spectrum, spec_extent = utils.process.calc_spectrum(raw_values, fs)
        filtered_values = utils.process.adaptive_bandpass_filter(raw_values, spectrum, fs)
        signal: pd.Series = pd.Series(filtered_values).apply(np.abs).rolling(72, center=True).mean()

        # deduce threshold0
        default_dash_duration = 1000  # in points
        window_size = 6 * default_dash_duration
        window_min = signal.rolling(window_size, center=True, min_periods=int(0.1 * window_size)).min()
        window_max = signal.rolling(window_size, center=True, min_periods=int(0.1 * window_size)).max()
        threshold0: pd.Series = (window_max + window_min) / 2

        # interval durations
        discrete0 = signal > threshold0
        nulls, beats = utils.process.split_sign_series(discrete0)
        null_info = utils.process.intervals_data(signal, nulls, np.min)
        beat_info = utils.process.intervals_data(signal, beats, np.max)

        # build threshold1
        peak_filtered = beat_info[beat_info['dur'] / fs > 0.01]
        low_filtered = null_info[null_info['dur'] / fs > 0.1]

        peak_points_x, peak_points_y = peak_filtered['p_mid'], peak_filtered['ampl_extr']
        peak_tck = interpolate.splrep(peak_points_x, peak_points_y)
        low_points_x, low_points_y = low_filtered['p_mid'], low_filtered['ampl_extr']
        low_tck = interpolate.splrep(low_points_x, low_points_y)

        def interp_peaks(n_point):
            return interpolate.splev(n_point, peak_tck)
        ip = np.vectorize(interp_peaks)

        def interp_nulls(n_point):
            return interpolate.splev(n_point, low_tck)
        il = np.vectorize(interp_nulls)

        tl = utils.process.time_labels_interval(wave_file, None, len(signal))
        max_inter = ip(np.arange(0, len(tl)))
        min_inter = il(np.arange(0, len(tl)))
        threshold1: pd.Series = min_inter + max_inter.max() / 8
        discrete1 = signal > threshold1

        _, beats2 = utils.process.split_sign_series(discrete1)
        beat_info2 = utils.process.intervals_data(signal, beats2, np.max)

        # classify intervals and decode morse
        di = beat_info2['dur']
        idi = di[di > 100]
        dots, dashes = idi[idi < idi.mean()], idi[idi > idi.mean()]
        reg = LinearRegression().fit(
            np.array(len(dots) * [1] + len(dashes) * [3]).reshape(-1, 1),
            np.concatenate([np.array(dots).astype(np.int), np.array(dashes).astype(np.int)])
        )
        dot_len = reg.predict([[1.]])[0]
        nnb_inter = utils.process.sign_series(discrete1)
        result = utils.process.predict(nnb_inter, dot_len)
        predictions[num] = result
        print(f'[{i + 1}/{len(file_numbers)}] {name} | {result}')
    return predictions

def main():
    with open(LABELS_CSV, 'r') as mlf:
        ldf = pd.read_csv(mlf, dtype={'ID': int, 'LABEL': str, 'MORSE': str})

    predictions = make_predictions(ldf['ID'].to_list())
    pdf = pd.DataFrame(predictions.items(), columns=['ID', 'PREDICTION'])

    elf = ldf[['ID', 'LABEL']].join(pdf.set_index('ID'), on='ID')
    elf.to_csv(PREDICTIONS_CSV, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
