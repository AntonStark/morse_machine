from matplotlib import pyplot as plt


def draw_time_series(*ys, wave_file, seconds, interactive, save_to):
    from utils.process import time_labels_interval
    for y in ys:
        t = time_labels_interval(wave_file, seconds, len(y))
        plt.plot(t, y)

    plt.gcf().set_size_inches(15, 5)
    if seconds:
        plt.gca().set_xlim(*seconds)

    if interactive or not save_to:
        plt.show()
    else:
        plt.savefig(save_to, dpi=100)
        plt.close()


def draw_time_series_with_points(*ys, beat_info, null_info,
                                 wave_file, fs, seconds, interactive, save_to):
    from utils.process import time_labels_interval
    for y in ys:
        t = time_labels_interval(wave_file, seconds, len(y))
        plt.plot(t, y)
    plt.plot(seconds[0] + (null_info['p_mid'] / fs), null_info['ampl_extr'], 'ko')
    plt.plot(seconds[0] + (beat_info['p_mid'] / fs), beat_info['ampl_extr'], 'r*')

    plt.gcf().set_size_inches(15, 5)
    plt.gca().set_xlim(*seconds)

    if interactive:
        plt.show()
    else:
        plt.savefig(save_to, dpi=100)
        plt.close()


def draw_hist(values, fs, title, interactive, save_to):
    if isinstance(values, list):
        to_hist = [(v / fs) for v in values]
    else:
        to_hist = values / fs

    plt.hist(to_hist, bins=100)
    plt.gca().set_title(title)
    if interactive or not save_to:
        plt.show()
    else:
        plt.savefig(save_to, dpi=100)
        plt.close()
