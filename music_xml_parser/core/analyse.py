import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pretty_midi

try:
    from plot import *
    from utils import midi2str, midi2pitchclass
except:

    from .plot import *
    from .utils import midi2str, midi2pitchclass

def getVoice(df_data: pd.DataFrame):
    v = df_data['Voice']
    return list(set(v))


def ambitus(df_data: pd.DataFrame):
    ab = []
    uni_parts = np.unique(df_data['PartID'].values)
    for i in uni_parts:
        d = df_data[df_data['PartID'].str.contains(i)].copy()
        d.dropna(subset=["MIDI"], inplace=True)
        max_r = np.max(d['MIDI'].to_numpy(dtype=float))
        min_r = np.min(d['MIDI'].to_numpy(dtype=float))
        diff_r = max_r-min_r
        ab.append([int(i), int(min_r), int(max_r), int(diff_r)])
    return ab


def pitch_histogram(df_data: pd.DataFrame, do_plot=True, visulize_midi_range=None):
    df_data.dropna(subset=["MIDI"], inplace=True)
    midi = df_data[['MIDI']].to_numpy()
    u,c = np.unique(midi, return_counts=True)
    if do_plot:
        barplot_pitch_histogram(u,counts=c, visulize_midi_range=visulize_midi_range)

    data = [[int(i), int(c)] for i, c in zip(u, c)]

    return data

def pitch_class_histogram(df_data: pd.DataFrame, do_plot=True):
    d = df_data.copy()
    d.dropna(subset=["Pitch"], inplace=True)
    d.drop(index=d[d['Pitch'] == 'rest'].index, inplace=True)
    d.drop_duplicates(subset='Pitch', keep="first", inplace=True)
    pitch_midi = d[['Pitch', 'MIDI']].to_numpy()
    dict_map = dict(pitch_midi)


    p_df = df_data.copy()
    d.dropna(subset=["Pitch"], inplace=True)
    p_df.drop(index=p_df[p_df['Pitch'] == 'rest'].index, inplace=True)
    p_o = p_df[['Pitch']].to_numpy()

    u, c = np.unique(p_o, return_counts=True)

    label_str =[]
    r_note =[]
    for i in u:
        l, rn = midi2pitchclass(dict_map[i])
        label_str.append(l)
        r_note.append(rn)

    if do_plot:
        barplot_pitch_class_histogram(r_note, c, label_str, x_axis_12pc=True)

    data = [[int(id), str(i), int(c)] for id, i, c in zip(r_note, label_str, c)]
    data.sort(key=lambda x: x[0])
    data = [[i[1], i[2]] for i in data]
    return data

def quarterlength_duration_histogram(df_data: pd.DataFrame, do_plot=True):
    dur = df_data['Duration'].to_numpy(dtype=float)
    u, c = np.unique(dur, return_counts=True)

    #a.sort(key=lambda x: x[1])
    labels = u
    if do_plot:
        barplot_quaterlength_duration_histogram(labels, counts=c)

    data = [[round(float(i),2), int(c)] for i, c in zip(u, c)]
    return data

def beatstrength(df_data: pd.DataFrame, do_plot_2D=True, do_plot_3D=False):


    pass
def interval(df_data: pd.DataFrame, part=None, do_plot=True):

    if part is None:
        part = 'all'
    if type(part) is str and  part != 'all':
        part = int(part)


    p_df1 = df_data.copy()
    p_df1.dropna(subset=["MIDI"], inplace=True)
    u_parts = np.unique(df_data['PartID'].to_numpy())
    u_parts = [int(i) for i in u_parts]
    if part in u_parts:pass
    elif part == None:pass
    elif part == 'all':pass
    else:
        raise Exception("Parts not found, give Valid Parts")

    if part == 'all':
        part_df = df_data.copy()
    elif part not in u_parts:
        grouped = p_df1.groupby(p_df1.PartID)
        part_df = grouped.get_group(str(part)).copy()
    else:
        part_df = df_data.copy()

    part_df.dropna(subset=["MIDI"], inplace=True)
    midi = part_df['MIDI'].to_numpy()
    diff = [int(t - s) for s, t in zip(midi, midi[1:])]

    labels, c = np.unique(diff, return_counts=True)
    if do_plot:
        barplot_intervals(labels, counts=c)
    data = [[int(i), int(c)] for i, c in zip(labels, c)]
    return data

def beat_strength(df_data: pd.DataFrame,with_pitch=False ,do_plot=True):
    df_data.dropna(subset=["MIDI"], inplace=True)
    df_data['beatstrength'] = pd.to_numeric(df_data['Offset']) - pd.to_numeric(df_data['Measure Offset'])
    if with_pitch==False:
        u, c = np.unique(df_data['beatstrength'].to_numpy(dtype=float), axis=0, return_counts=True)

        if do_plot:
            barplot(u, counts=c, x_label='Beat Strength', y_label='Occurrences')
        data = [[int(i), int(c)] for i, c in zip(u, c)]
        return data
    else:
        n_df = df_data[['MIDI', 'beatstrength']].to_numpy(dtype=float)
        u,c  = np.unique(n_df, axis=0, return_counts=True)

        p = [midi2str(i) for i in u[:,0]]

        data = np.array([p, u[:,1], c]).T

        if do_plot:
            beat_stength_3d(data)
        return data






if __name__=='__main__':
    import sys
    import os

    sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))
    import hfm.scripts_in_progress.xml_parser.music_xml_parser as mp

    xml_file = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'
    # xml_file = 'PrJode_Jos1102_COM_2-5_MissaLasol_002_00138.xml'


    # m_df, meaure_onset = mp.parse.with_xml_file(file_name=xml_file,
    #                               plot_pianoroll=False,
    #                               save_at=None,
    #                               save_file_name=None,
    #                               do_save=False,
    #                               x_axis_res=2, get_measure_onset=True)


    m_df = mp.parse.with_xml_file(file_name=xml_file,
                                  plot_pianoroll=True,
                                  save_at=None,
                                  save_file_name=None,
                                  do_save=False,
                                  x_axis_res=2, get_measure_onset=False)
    # print(m_df)
    # data = beat_strength(df_data=m_df,do_plot=True)
