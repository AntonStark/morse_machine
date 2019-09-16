import numpy as np
from wave import open as open_wave


def load_raw(filepath, seconds=None):
    wave_file = open_wave(filepath, 'rb')
    nframes, framerate = wave_file.getnframes(), wave_file.getframerate()
    if seconds:
        start, end = (2 * s * framerate for s in seconds)  # "2" because of frame size
        if end > 2 * nframes:
            raise IndexError(
                f'file {filepath} duration: {nframes / framerate}s., interval=({seconds[0]}, {seconds[1]}) given')
        wav_frames = wave_file.readframes(nframes)[int(start):int(end)]
    else:
        wav_frames = wave_file.readframes(nframes)
    ys = np.frombuffer(wav_frames, dtype=np.int16)
    return ys, wave_file
