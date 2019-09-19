import numpy as np
import wave
from matplotlib import mlab
from scipy.signal import butter, sosfiltfilt


#
# WAV FILE STUFF
#

def load_raw(filepath, seconds=None):
    wave_file = wave.open(filepath, 'rb')
    nframes, framerate = wave_file.getnframes(), wave_file.getframerate()
    if seconds:
        start, end = (2 * s * framerate for s in seconds)  # "2" because of frame size
        if end > 2 * nframes:
            raise IndexError(
                f'file {filepath} duration: {nframes / framerate}s., interval=({seconds[0]}, {seconds[1]}) given')
        wav_frames = wave_file.readframes(nframes)[int(start):int(end)]
    else:
        wav_frames = wave_file.readframes(nframes)
    wave_file.close()
    ys = np.frombuffer(wav_frames, dtype=np.int16)
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
