"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""

try:
    from .parser_utils import ZoomPan
except:
    from parser_utils import ZoomPan

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.patches import Rectangle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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

def _get_midi_mapping_dict():
    s_21 = ['na' for i in range(21)]
    s_19 = ['na' for i in range(19)]
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
        df, do_plot, measure_duration_list, x_axis_res, get_measure_onset, measure_offset_data = func(*args, **kwargs)

        measure = m_dur_off(measure_duration_list)

        if do_plot:
            offset = list(np.squeeze(df['Offset'].to_numpy(dtype=float)))
            duration = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
            total_measure = int(max(list(np.squeeze(df['Measure'].to_numpy(dtype=float)))))
            measure = measure[:total_measure]
            midi = df['MIDI'].replace({np.nan: 0}).to_list()
            partid = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))
            part_name = list(np.squeeze(df['Part Name'].to_numpy(dtype=str)))
            _create_pianoroll_single_parts(pitch=midi, time=offset, measure=measure, partid=partid,part_name=part_name, duration=duration,
                                           midi_min=55, midi_max=75, x_axis_res=x_axis_res)
        if get_measure_onset:
            return df, measure_offset_data
        else:
            return df
    return plotting_wrapper_parts

def _get_xtickslabels_with_measure(x_axis, measure):
    x_lab = []
    for i in range(len(x_axis)):
        s = x_axis[i]
        idx = np.argmin(np.abs(s-measure))

        if s == measure[idx]:
            l = '\n\n'+str(measure.index(measure[idx]))
        else:
            l =''
        x_lab.append(str(s)+l)
    return x_lab

def _create_pianoroll_single_parts(pitch, time, measure, partid, part_name, duration,
                                   midi_min, midi_max, x_axis_res, *args, **kwargs):
    cm = plt.get_cmap('gist_rainbow')

    x_axis = np.arange(0, max(time) * x_axis_res + 1) / x_axis_res
    NUM_PARTS = len(list(set(partid)))
    colors = [cm(1. * i / NUM_PARTS) for i in range(NUM_PARTS)]

    labels_128 = _get_midi_labels_128()
    s_part_names = list(set(part_name))
    assert len(s_part_names) == NUM_PARTS
    labels_set = s_part_names
    #labels_set = [str(s_part_names[i-1]) + str(i) for i, pn in range(1, NUM_PARTS + 1)]


    colors_dicts = {}

    for i, l in enumerate(labels_set):
        colors_dicts[l] = colors[i]

    assert np.shape(pitch)[0] == np.shape(time)[0]
    f = plt.figure(figsize=(16, 9))
    ax = f.add_subplot(111)

    for i in range(np.shape(time)[0]):
        t = time[i]
        color_prt = colors[partid[i] - 1]
        c_d = duration[i]

        if pitch[i] == 0:
            continue
        else:
            p = int(pitch[i])
            a = 0.5
        ax.add_patch(Rectangle((t, p - 0.5), width=c_d, height=1, edgecolor='k', facecolor=color_prt, fill=True))
    for tt in measure:
        ax.vlines(tt, ymax=500, ymin=0, colors='grey', linestyles=(0, (2, 15)))

    xlab = _get_xtickslabels_with_measure(x_axis, measure)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(xlab)

    ax.set_yticks(np.arange(128))
    ax.set_yticklabels(labels_128)

    p = [i for i in pitch if i != 0]
    ax.set_ylim([min(p) - 1.5, max(p) + 1.5])

    ax.set_xlim([0, int(x_axis[-1]*0.20)])
    print(int(x_axis[-1]*0.20))
    # ax.set_xlim([0, 5])

    ax.set_xlabel("Time \n Measure Number")
    ax.set_ylabel("Pitch")

    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors_dicts[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)

    zp = ZoomPan()
    _ = zp.zoom_factory(ax, base_scale=1.1)
    _ =zp.pan_factory(ax)
    plt.show()


def barplot_pitch_histogram(labels, counts, visulize_midi_range=None):
    try:
        from utils import midi2str
    except:
        from .utils import midi2str


    if visulize_midi_range is None:
        visulize_midi_range = [min(labels)-0.5, max(labels)+0.5]

    midi_labels = [midi2str(i) for i in range(128)]
    f = plt.figure(figsize=(12, 4))

    ax = f.add_subplot(111)

    ax.bar(labels, counts, width=0.4, color='darkslateblue', alpha=0.8)
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Occurrences')
    ax.set_xticks(np.arange(128))


    ax.set_xticklabels(midi_labels)
    ax.set_xlim(visulize_midi_range)
    plt.grid()
    plt.show()

def barplot_pitch_class_histogram(labels, counts, label_str, x_axis_12pc =False):

    f = plt.figure(figsize=(12, 4))
    ax = f.add_subplot(111)
    ax.bar(labels, counts, width=0.4, color='darkslateblue', alpha=0.8)
    ax.set_xlabel('Pitch')
    ax.set_ylabel('Occurrences')
    if x_axis_12pc:
        ax.set_xticks(np.arange(12))
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        ax.set_xticklabels(names)
    else:
        ax.set_xticklabels(label_str)

    plt.grid()
    plt.show()
def barplot_quaterlength_duration_histogram(labels, counts):

    f = plt.figure(figsize=(12, 4))
    ax = f.add_subplot(111)
    ax.bar(labels, counts, width=0.4, color='darkslateblue', alpha=0.8)
    ax.set_xlabel('Quater Length')
    ax.set_ylabel('Occurrences')
    ax.set_xticks(np.arange(np.max(labels)+1))

    plt.grid()
    plt.show()

def barplot_intervals(labels, counts):

    f = plt.figure(figsize=(12, 4))
    ax = f.add_subplot(111)
    ax.bar(labels, counts, width=0.4, color='darkslateblue', alpha=0.8)
    ax.set_xlabel('Intervals')
    ax.set_ylabel('Occurrences')
    ax.set_xticks(np.arange(np.min(labels), np.max(labels)+1))

    plt.grid()
    plt.show()

def barplot(labels, counts, x_label='x_label', y_label='y_label'):

    f = plt.figure(figsize=(12, 4))
    ax = f.add_subplot(111)
    ax.bar(labels, counts, width=0.2, color='darkslateblue', alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_xticks(np.arange(np.min(labels), np.max(labels)+1))

    plt.grid()
    plt.show()

def beat_stength_3d(np_bs_data):
    n_uni = np.unique(np_bs_data[:, 0])
    bs_uni = np.unique(np_bs_data[:, 1])
    n_uni_int_dict = dict(zip(n_uni, np.arange(len(n_uni))))
    bs_uni_int_dict = dict(zip(bs_uni, np.arange(len(bs_uni))))
    plt_bs_data = np.zeros(np.shape(np_bs_data))
    for i in range(np.shape(np_bs_data)[0]):
        plt_bs_data[i][0] = n_uni_int_dict[np_bs_data[i][0]]
        plt_bs_data[i][1] = bs_uni_int_dict[np_bs_data[i][1]]
        plt_bs_data[i][2] = np_bs_data[i][2]
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(111, projection='3d')
    numele = np.shape(np_bs_data)[0]
    x = plt_bs_data[:, 0]
    y = plt_bs_data[:, 1]
    z = np.zeros(numele)

    dx = 0.5 * np.ones(numele)
    dy = 0.3 * np.ones(numele)
    dz = plt_bs_data[:, 2]

    cmap = cm.get_cmap('jet')  # Get desired colormap
    max_height = np.max(dz)  # get range of colorbars
    min_height = np.min(dz)
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax1.set_xticks(np.arange(len(n_uni)))
    ax1.set_yticks(np.arange(len(bs_uni)))

    ax1.bar3d(x, y, z, dx, dy, dz, color=rgba, zsort='average')

    ax1.set_xticklabels(list(n_uni))
    ax1.set_yticklabels([str(i) for i in bs_uni])
    ax1.set_xlabel('Notes')
    ax1.set_ylabel('Beat Strength')
    ax1.set_zlabel('Occurrence')
    plt.grid()
    plt.show()