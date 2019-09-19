from . import process
from . import plot


def audio_filepath(audio_dir, name):
    import os
    return os.path.join(audio_dir, name)


def plot_filepath(plots_dir, name):
    import os
    return os.path.join(plots_dir, name)
