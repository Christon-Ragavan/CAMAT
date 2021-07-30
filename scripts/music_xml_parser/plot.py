import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import ZoomPan


def painoroll_partlevel():
    pass


def plotting_wrapper_parts(df):

    def wrap(func):

        offset = list(np.squeeze(df['Offset'].to_numpy(dtype=float)))
        duration = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        midi = list(np.squeeze(df['MIDI'].to_numpy(dtype=int)))
        measure = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        partid = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))
        _create_pianoroll_single_parts(pitch=midi, time=offset, measure=measure, partid=partid,duration =duration, midi_min=55, midi_max=75)
    return wrap()



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


