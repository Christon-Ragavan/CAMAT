from matplotlib import colors
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import pandas as pd


from zoom_pan import ZoomPan
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def _demo():
    time = 50
    pitch = 12
    mid_pianoroll = np.zeros((pitch,time))
    print(np.shape(mid_pianoroll))
    chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for i in range(time):
        p_idx = int(np.random.randint(low=0, high=12, size=1))
        mid_pianoroll[p_idx,i] = 1
        if p_idx >6:
            mid_pianoroll[p_idx, i] = 2
    eval_cmap = colors.ListedColormap([[1,1,1], "k","blue"])

    # plt.imshow(mid_pianoroll, origin='lower', aspect='auto', cmap='gray_r')
    plt.imshow(mid_pianoroll, origin='lower', aspect='auto', cmap=eval_cmap)
    plt.yticks(ticks=np.arange(pitch), labels= chroma_label)
    plt.xlabel('Time (Not Defined)')
    plt.ylabel('Piano Roll (Chroma Pitch')
    plt.tight_layout()
    plt.show()


def compute_visualization_array(annotations, analysis):
    true_positives = annotations * analysis

    false_positives = analysis - true_positives
    false_negatives = annotations - true_positives
    results = 3 * true_positives + 2 * false_negatives + 1 * false_positives

    return results


def plot_eval_matrix(annotations, analysis, Fs=1, Fs_F=1, xlabel='Time (seconds)', ylabel='', title='', clim=[0, 4],
                     ax=None):
    X = compute_visualization_array(annotations=annotations, analysis=analysis)
    eval_cmap = colors.ListedColormap([[1, 1, 1], [1, 0.3, 0.3], [1, 0.7, 0.7], [0, 0, 0]])
    eval_bounds = np.array([0, 1, 2, 3, 4]) - 0.5
    eval_norm = colors.BoundaryNorm(eval_bounds, 4)
    eval_ticks = [0, 1, 2, 3]

    T_coef = np.arange(X.shape[1]) / Fs
    F_coef = np.arange(X.shape[0]) / Fs_F
    x_ext1 = (T_coef[1] - T_coef[0]) / 2
    x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
    y_ext1 = (F_coef[1] - F_coef[0]) / 2
    y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
    extent = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]

    im = ax[0].imshow(X, origin='lower', aspect='auto', cmap=eval_cmap, norm=eval_norm, extent=extent)
    cbar = plt.colorbar(im, cax=ax[1], cmap=eval_cmap, norm=eval_norm, boundaries=eval_bounds, ticks=eval_ticks)
    cbar.ax.set_yticklabels(['', 'FP', 'FN', 'TP'])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)

    return im

def _extract_time_midipitch(pitch, time):

    pass


def plot_piano_roll(pitch, time, octave_range=None, xlabel='Time (seconds)', ylabel='Octaves', title='PianoRoll'):
    """

    :param data:
    :param octave_range: (str)
                        C4-C5
    :param xlabel:
    :param ylabel:
    :param title:
    :return:c
    """
    # pitch, time = _extract_time_midipitch(pitch, time)
    pitch_c = []

    for i in pitch:
        if i == np.inf:
            pitch_c.append(0)
        else:
            pitch_c.append(i)
    print(type(pitch), type(time), np.shape(pitch), np.shape(time))
    fig, ax = plt.subplots()
    ax.plot(time, pitch_c)

    # zp = ZoomPan()
    # figZoom = zp.zoom_factory(ax, base_scale=.2)
    # figPan = zp.pan_factory(ax)
    # show()
    plt.show()

    # time_bins = np.shape(time)[0]
    # mid_pianoroll = np.zeros((13,time_bins))
    # print("mid_pianoroll", mid_pianoroll)
    # n_pitch = []
    #
    # for i in pitch:
    #     if type(i)== str:
    #         print(i, type(i))
    #         n_pitch.append(0)
    #     else:
    #         n_pitch.append(int(i))
    # if octave_range==None:
    #     n_pitch = [int(i%12) for i in n_pitch]
    #
    #
    # for i in range(time_bins):
    #     p_idx = n_pitch[i]
    #     print("pitch {} time {} time_idx {}".format(p_idx, time[i],i))
    #
    #     print("p_idx", p_idx)
    #     mid_pianoroll[p_idx,i] = 1
    #     if p_idx ==0:
    #         mid_pianoroll[p_idx, i] = 2
    #
    # # print(pitch)
    # print(n_pitch)
    # # plt.plot(time, n_pitch)
    # # plt.show()
    # chroma_label = ['NA','C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # eval_cmap = colors.ListedColormap([[1,1,1], "k","blue"])
    #
    # plt.imshow(mid_pianoroll, origin='lower', aspect='auto', cmap=eval_cmap)
    # plt.yticks(ticks=np.arange(13), labels=chroma_label)
    # plt.xlabel('Time (Not Defined)')
    # plt.ylabel('Piano Roll (Chroma Pitch')
    # plt.tight_layout()
    # plt.show()
def _get_midi_labels_128():
    s_21 = ['-'for i in range(21)]
    s_19 = ['-'for i in range(19)]
    chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    s_m = ['A0', 'A#0', 'B0']
    midi_labels_128 = []
    midi_labels_128.extend(s_21)
    midi_labels_128.extend(s_m)

    for o in range(1,8):
        for p in chroma_label:
            midi_labels_128.append(p+str(o))
    midi_labels_128.append('C8')
    midi_labels_128.extend(s_19)

    # for i in midi_labels_128:
    #     print(i)
    # print(len(midi_labels_128))
    assert len(midi_labels_128)==128
    return midi_labels_128


def _create_pianoroll( data, midi_min, midi_max):
    pitch = data["MIDI Pitch"].to_numpy()
    time = data["Start"].to_numpy()
    instrument = data["Instrument"].to_numpy()
    duration = data["Duration"].to_numpy()
    Measure_part1 = data["Measure_part1"].to_numpy()

    Measure_l = list(Measure_part1)

    pitch = [ 0 if i == np.inf else i  for i in pitch]

    num_inst = np.sort(list(set(instrument)))
    colors = cm.rainbow(np.linspace(0, 1, len(num_inst)))
    inst_dict = {inst_c:colors[i] for i, inst_c in enumerate (num_inst)}

    labels_128 =_get_midi_labels_128()
    assert np.shape(pitch) [0]== np.shape(time)[0]
    time_axis = np.arange(0, time[-1],step=0.10)
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(1, 1, 1)

    n_measure = list(np.sort(list(set(Measure_part1))))

    for m in n_measure:
        mp = int(Measure_l.index(m))
        ax.axvline(x=time[mp], linewidth=1, color='k', alpha=0.5, linestyle='--')

    for i in range(np.shape(time)[0]):
        fc = inst_dict[instrument[i]]
        if pitch[i]== 0:
            continue
        else:
            p = int(pitch[i])
            a = 0.6
        t = time[i]
        d = duration[i]

        ax.add_patch(Rectangle((t, p-0.5), d, 1, edgecolor='k', facecolor=fc, fill=True, alpha=a))
    ax.set_yticks(np.arange(128))
    ax.set_yticklabels(labels_128)
    # ax.set_xticks(np.arange(time))

    ax.set_xticklabels(Measure_part1)
    p = []

    for i in pitch:
        if i !=0:
            p.append(i)

    ax.set_ylim([min(p) - 1.5, max(p) + 1.5])

    ax.set_xlim([0, 45])



    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale=1.1)
    figPan = zp.pan_factory(ax)

    plt.show()




def _create_pianoroll_single(pitch, time,duration, midi_min, midi_max):
    pitch = [ 0 if i == np.nan else i  for i in pitch]

    labels_128 =_get_midi_labels_128()
    assert np.shape(pitch) [0]== np.shape(time)[0]
    time_axis = np.arange(0, time[-1],step=0.10)
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(1, 1, 1)

    for i in range(np.shape(time)[0]):
        c_d = duration[i]
        if pitch[i]== 0:
            continue
        else:
            p = int(pitch[i])
            a = 0.6
        t = time[i]

        ax.add_patch(Rectangle((t, p-0.1), width=c_d, height= 0.2, edgecolor='k', facecolor='red', fill=True, alpha=a))
    ax.set_yticks(np.arange(128))
    ax.set_yticklabels(labels_128)
    p = []
    for i in pitch:
        if i !=0:
            p.append(i)
    ax.set_ylim([min(p) - 1.5, max(p) + 1.5])

    ax.set_xlim([0, 20])
    ax.set_xlabel("Offset")
    ax.set_ylabel("Pitch")


    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale=1.1)
    figPan = zp.pan_factory(ax)

    plt.show()


def _create_sparse_rep(dlist):
    n_sparse = []
    for i, m in enumerate(dlist):
        if i == 0:
            n_sparse.append(1)
        else:
            if m !=dlist[i-1]:
                n_sparse.append(1)
            else:
                n_sparse.append(0)
    return n_sparse



def _create_pianoroll_single_parts(pitch, time, measure, partid, duration, midi_min, midi_max):
    pitch = [ 0 if i == np.nan else i  for i in pitch]
    cm = plt.get_cmap('gist_rainbow')

    NUM_COLORS = 4
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    measure_s = _create_sparse_rep(list(measure))
    partid_s = _create_sparse_rep(list(partid))
    labels_128 =_get_midi_labels_128()
    assert np.shape(pitch) [0]== np.shape(time)[0]
    time_axis = np.arange(0, time[-1],step=0.10)
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(1, 1, 1)
    #colors = ['r', 'g', 'b', 'k']

    for i in range(np.shape(time)[0]):
        t = time[i]
        if measure_s[i]==1:
            #ax.vlines(t,ymax=500, ymin=0, colors='k', linestyles='dotted')
            ax.vlines(t, ymax=500, ymin=0, colors='grey', linestyles=(0,(2,15)))

        color_prt = colors[partid[i]-1]
        c_d = duration[i]
        if pitch[i]== 0:
            continue
        else:
            p = int(pitch[i])
            a = 0.6

        ax.add_patch(Rectangle((t, p-0.3), width=c_d, height= 0.6, edgecolor='k', facecolor=color_prt, fill=True, alpha=a))

    ax.set_yticks(np.arange(128))
    ax.set_yticklabels(labels_128)
    p = []
    for i in pitch:
        if i !=0:
            p.append(i)
    ax.set_ylim([min(p) - 1.5, max(p) + 1.5])

    ax.set_xlim([0, 20])
    ax.set_xlabel("Offset")
    ax.set_ylabel("Pitch")


    zp = ZoomPan()
    figZoom = zp.zoom_factory(ax, base_scale=1.1)
    figPan = zp.pan_factory(ax)

    plt.show()




def plot_piano_roll02(pitch, time, octave_range=None, xlabel='Time (seconds)', ylabel='Octaves', title='PianoRoll'):
    """

    :param data:
    :param octave_range: (str)
                        C4-C5
    :param xlabel:
    :param ylabel:
    :param title:
    :return:c
    """
    time_bins = np.shape(time)[0]
    mid_pianoroll = np.zeros((13,time_bins))
    print("mid_pianoroll", mid_pianoroll)
    n_pitch = []

    for i in pitch:
        if type(i)== str:
            print(i, type(i))
            n_pitch.append(0)
        else:
            n_pitch.append(int(i))
    if octave_range==None:
        n_pitch = [int(i%12) for i in n_pitch]


    for i in range(time_bins):
        p_idx = n_pitch[i]
        print("p_idx", p_idx)
        mid_pianoroll[p_idx,i] = 1
        if p_idx ==0:
            mid_pianoroll[p_idx, i] = 2

    # print(pitch)
    print(n_pitch)
    # plt.plot(time, n_pitch)
    # plt.show()
    chroma_label = ['NA','C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    eval_cmap = colors.ListedColormap([[1,1,1], "k","blue"])

    plt.imshow(mid_pianoroll, origin='lower', aspect='auto', cmap=eval_cmap)
    plt.yticks(ticks=np.arange(13), labels=chroma_label)
    plt.xlabel('Time (Not Defined)')
    plt.ylabel('Piano Roll (Chroma Pitch')
    plt.tight_layout()
    plt.show()



def testing():
    a = np.array([[1, 2], [2, 4], [4, 6]])
    plt.imshow(a.T, cmap="gray_r")
    plt.show()

