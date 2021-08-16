import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from plot import *
except:

    from .plot import *

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
    #p_o = df_data[['Pitch', 'Octave']].to_numpy()
    df_data.dropna(subset=["MIDI"], inplace=True)
    p_o = df_data[['MIDI']].to_numpy()
    u,c = np.unique(p_o, return_counts=True)
    labels = u
    if do_plot:
        barplot_pitch_histogram(labels,counts=c, visulize_midi_range=visulize_midi_range)

    data = [[int(i), int(c)] for i, c in zip(u, c)]

    return data

def pitch_class_histogram(df_data: pd.DataFrame, do_plot=True):
    d = df_data[df_data['Pitch'].str.contains('rest')].copy()
    d.dropna(subset=["Pitch"], inplace=True)
    p_o = df_data[['Pitch']].to_numpy()
    u, c = np.unique(p_o, return_counts=True)
    labels = u
    if do_plot:
        barplot_pitch_class_histogram(labels, counts=c)

    data = [[str(i), int(c)] for i, c in zip(u, c)]
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


if __name__=='__main__':
    import sys
    import os

    sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))
    import hfm.scripts_in_progress.xml_parser.music_xml_parser as mp

    xml_file = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'


    # m_df, meaure_onset = mp.parse.with_xml_file(file_name=xml_file,
    #                               plot_pianoroll=False,
    #                               save_at=None,
    #                               save_file_name=None,
    #                               do_save=False,
    #                               x_axis_res=2, get_measure_onset=True)

    m_df = mp.parse.with_xml_file(file_name=xml_file,
                                  plot_pianoroll=False,
                                  save_at=None,
                                  save_file_name=None,
                                  do_save=False,
                                  x_axis_res=2, get_measure_onset=False)




    # beatstrength(m_df, True, False)