import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import ZoomPan

from matplotlib import colors
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def painoroll_partlevel():
    pass


def _get_midi_labels_128() -> list:
    s_21 = ['-' for i in range(21)]
    s_19 = ['-' for i in range(19)]
    chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    s_m = ['A0', 'A#0', 'B0']
    midi_labels_128 = []
    midi_labels_128.extend(s_21)
    midi_labels_128.extend(s_m)
    for o in range(1, 8):
        for p in chroma_label:
            midi_labels_128.append(p + str(o))
    midi_labels_128.append('C8')
    midi_labels_128.extend(s_19)
    assert len(midi_labels_128) == 128
    return midi_labels_128


def _create_sparse_rep(dlist):
    n_sparse = []
    for i, m in enumerate(dlist):
        if i == 0:
            n_sparse.append(1)
        else:
            if m != dlist[i - 1]:
                n_sparse.append(1)
            else:
                n_sparse.append(0)
    return n_sparse


def pianoroll_parts(func, *args, **kwargs):
    def m_dur_off(m_d):
        s = 0.0
        m_o = []
        for i in m_d:
            s += float(i)
            m_o.append(s)
        return m_o

    def plotting_wrapper_parts(*args, **kwargs):
        df, do_plot, measure_duration_list = func(*args, **kwargs)
        measure = m_dur_off(measure_duration_list)
        if do_plot:
            offset = list(np.squeeze(df['Offset'].to_numpy(dtype=float)))
            duration = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
            midi = list(np.squeeze(df['MIDI'].to_numpy(dtype=int)))
            partid = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))
            _create_pianoroll_single_parts(pitch=midi, time=offset, measure=measure, partid=partid, duration=duration,
                                           midi_min=55, midi_max=75)
        return df

    return plotting_wrapper_parts


def _create_pianoroll_single_parts(pitch, time, measure, partid, duration,
                                   midi_min, midi_max, *args, **kwargs):
    pitch = [0 if i == np.nan else i for i in pitch]
    cm = plt.get_cmap('gist_rainbow')

    NUM_COLORS = 4
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    #measure_s = _create_sparse_rep(list(measure))
    labels_128 = _get_midi_labels_128()
    assert np.shape(pitch)[0] == np.shape(time)[0]
    time_axis = np.arange(0, time[-1], step=0.10)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)


    for i in range(np.shape(time)[0]):
        t = time[i]

        color_prt = colors[partid[i] - 1]
        c_d = duration[i]
        if pitch[i] == 0:
            continue
        else:
            p = int(pitch[i])
            a = 0.6

        ax.add_patch(
            Rectangle((t, p - 0.3), width=c_d, height=0.6, edgecolor='k', facecolor=color_prt, fill=True, alpha=a))
    for tt in measure:
        ax.vlines(tt, ymax=500, ymin=0, colors='grey', linestyles=(0, (2, 15)))

    ax.set_yticks(np.arange(128))
    ax.set_yticklabels(labels_128)
    p = []
    for i in pitch:
        if i != 0:
            p.append(i)
    ax.set_ylim([min(p) - 1.5, max(p) + 1.5])

    ax.set_xlim([0, 20])
    ax.set_xlabel("Offset")
    ax.set_ylabel("Pitch")

    # print(t_mea)
    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale=1.1)
    figPan = zp.pan_factory(ax)
    plt.show()
