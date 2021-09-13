import re

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

def search_upeat_files():
    pass


def max_measure_num(df_data: pd.DataFrame, part='all'):
    df_c = df_data.copy()
    df_c.drop_duplicates(subset='PartID', keep="last", inplace=True)
    df_c_n = df_c[['Measure', 'PartID', 'Part Name']].to_numpy()

    return df_c_n

def metric_profile_split_time_signature(df_data: pd.DataFrame,
                                       with_pitch=False,
                                       do_plot=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    ts = df_data['Time Signature'].to_numpy()
    u, c = np.unique(ts, return_counts=True)
    pd_list = []
    for ts_c in u:
        c_d = df_data.loc[df_data['Time Signature'] == ts_c].copy()
        curr_h = metric_profile(c_d, x_label=f"Metric Profile (Time Signature : {ts_c})",
                               with_pitch=with_pitch,
                               do_plot=do_plot)
        pd_list.append(curr_h)
    return pd_list

def duration_histogram(df_data: pd.DataFrame,
                       with_pitch=False,
                       do_plot=True):
    pass

def time_signature_histogram(df_data: pd.DataFrame, do_plot=False, do_adjusted=False, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    if not do_adjusted:
        ts = df_data['Time Signature'].to_numpy()
        xlab = 'Time Signature'

    else:
        xlab = 'Time Signature Adjusted'
        ts = df_data['Time Signature Adjusted'].to_numpy()

    u, c = np.unique(ts, return_counts=True)
    if do_plot:
        barplot(u, counts=c, figsize='fit', x_label=xlab, y_label='Occurrences')
    data = [[i, int(c)] for i, c in zip(u, c)]
    data.sort(key=lambda x: x[1])

    return data


def ambitus(df_data: pd.DataFrame, output_as_midi=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)

    ab = []
    uni_parts = np.unique(df_data['PartID'].values)
    for i in uni_parts:
        d = df_data[df_data['PartID'].str.contains(i)].copy()
        d.dropna(subset=["MIDI"], inplace=True)
        max_r = np.max(d['MIDI'].to_numpy(dtype=float))
        min_r = np.min(d['MIDI'].to_numpy(dtype=float))
        diff_r = max_r - min_r

        if output_as_midi:
            ab.append([int(i), int(min_r), int(max_r), int(diff_r)])
        else:

            min_r = midi2str(int(min_r))
            max_r = midi2str(int(max_r))
            ab.append([int(i), str(min_r), str(max_r), int(diff_r)])

    return ab


def pitch_histogram(df_data: pd.DataFrame, do_plot=True, do_plot_full_axis=True, visulize_midi_range=None, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    df_data.dropna(subset=["MIDI"], inplace=True)
    midi = df_data[['MIDI']].to_numpy()
    u, c = np.unique(midi, return_counts=True)
    if do_plot:
        barplot_pitch_histogram(u,
                                c,
                                do_plot_full_axis,
                                visulize_midi_range=visulize_midi_range)
    pitch_str = [midi2str(int(i)) for i in u]
    data = [[int(i), str(p), int(c)] for i,p, c in zip(u, pitch_str, c)]

    return data


def pitch_class_histogram(df_data: pd.DataFrame, x_axis_12pc=True, do_plot=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
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

    label_str = []
    r_note = []
    for i in u:
        l, rn = midi2pitchclass(dict_map[i])
        label_str.append(l)
        r_note.append(rn)

    if do_plot:
        barplot_pitch_class_histogram(r_note, c, label_str, x_axis_12pc=x_axis_12pc)

    data = [[int(id), str(i), int(c)] for id, i, c in zip(r_note, label_str, c)]
    data.sort(key=lambda x: x[0])
    data = [[i[1], i[2]] for i in data]
    return data


def quarterlength_duration_histogram(df_data: pd.DataFrame,
                                     with_pitch=False,
                                     with_pitchclass=False,
                                     do_plot=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)


    if not with_pitch:
        dur = df_data['Duration'].to_numpy(dtype=float)
        u, c = np.unique(dur, return_counts=True)
        # a.sort(key=lambda x: x[1])
        labels = u
        if do_plot:
            barplot_quaterlength_duration_histogram(labels, counts=c)

        data = [[round(float(i), 2), int(c)] for i, c in zip(u, c)]
    else:
        df_data.dropna(subset=["MIDI"], inplace=True)

        n_df = df_data[['MIDI', 'Duration']].to_numpy(dtype=float)
        u, c = np.unique(n_df, axis=0, return_counts=True)



        p = [int(i) for i in u[:, 0]]
        d = [float(i) for i in u[:, 1]]

        pd_data_s = pd.DataFrame(np.array([p, d, c]).T, columns=['Pitch', 'Duration', 'Count'])
        convert_dict = {'Count': int,
                        'Duration': float
                        }
        pd_data_s = pd_data_s.astype(convert_dict)
        data = pd_data_s.to_numpy()
        if do_plot:
            plot_3d(np.array(data))
    return data


def interval(df_data: pd.DataFrame, part=None, do_plot=True,filter_dict=None):
    # v = df_data[['PartID', 'Part Name']].drop_duplicates().to_numpy()
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    if part is None:
        part = 'all'
    if type(part) is str and part != 'all':
        part = int(part)

    p_df1 = df_data.copy()
    p_df1.dropna(subset=["MIDI"], inplace=True)
    u_parts = np.unique(df_data['PartID'].to_numpy())
    u_parts = [int(i) for i in u_parts]
    if part in u_parts:
        pass
    elif part == None:
        pass
    elif part == 'all':
        pass
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


def metric_profile(df_data: pd.DataFrame,
                  x_label='Metric Profile',
                  with_pitch=False,
                  do_plot=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)

    df_data.dropna(subset=["MIDI"], inplace=True)
    df_data['metricprofile'] = pd.to_numeric(df_data['Offset']) - pd.to_numeric(df_data['Measure Offset'])
    if with_pitch == False:
        u, c = np.unique(df_data['metricprofile'].to_numpy(dtype=float), axis=0, return_counts=True)
        u = [i+1for i in u]
        if do_plot:
            barplot_mp(u, counts=c, x_label=x_label, y_label='Occurrences')
        data = [[int(i), int(c)] for i, c in zip(u, c)]
        return data
    else:
        n_df = df_data[['MIDI', 'metricprofile']].to_numpy(dtype=float)
        u, c = np.unique(n_df, axis=0, return_counts=True)
        p = [int(i) for i in u[:, 0]]
        pitch = [midi2str(int(i)) for i in u[:, 0]]

        pd_data_s = pd.DataFrame(np.array([p, u[:, 1], c]).T, columns=['Pitch', 'metricprofile', 'Count'])
        convert_dict = {'Count': int,'metricprofile': float }
        pd_data_s = pd_data_s.astype(convert_dict)
        data = pd_data_s.to_numpy()


        if do_plot:
            beat_stength_3d(data, ylabel='Metric Profile')
        data_f = pd.DataFrame(np.array([p, pitch, u[:, 1], c]).T, columns=['MIDI', 'Pitch', 'metricprofile', 'Count'])
        convert_dict_2 = {'Count': int,'metricprofile': float }
        data_f = data_f.astype(convert_dict_2)
        data_2 = data_f.to_numpy()

        return data_2



def filter(df_data: pd.DataFrame, filter_dict):
    """
    Order of the dicture is important
    :param df_data:
    :param filter_dict:
    :return:
    """
    f_df = df_data.copy()

    for i in filter_dict:
        s_d = str(filter_dict[i])
        if '-' in s_d:
            s,e = re.split('-',s_d, 2)
            arr = np.arange(int(s), int(e)+1)
            df_list = []
            for ii in arr:
                grouped = f_df.groupby(by=[i])
                df_list.append(grouped.get_group(str(ii)).copy())
            f_df = pd.concat(df_list,
                                ignore_index=True,
                                verify_integrity=False,
                                copy=False)
        else:
            grouped = f_df.groupby(by=[i])
            f_df = grouped.get_group(s_d).copy()

    return f_df


if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))
    import hfm.scripts_in_progress.xml_parser.music_xml_parser as mp

    # xml_file = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'
    # xml_file = 'PrJode_Jos1102_COM_2-5_MissaLasol_002_00138.xml'
    # xml_file = 'BeLuva_Op59_1-3_1-4_StringQuar_003_00129.xml'

    # filter_dict_t = {'Measure': '6-7', 'PartID': '2-3'}

    xml_file = 'MoWo_K279_COM_1-3_PianoSonat_003_00920.xml'
    m_df = mp.parse.with_xml_file(file_name=xml_file,
                                plot_pianoroll=True,
                                plot_inline_ipynb=True,
                                save_at=None,
                                save_file_name=None,
                                do_save=False,
                                x_axis_res=2,
                                get_measure_onset=False)
    print(m_df)
    # m_df = mp.parse.with_xml_file(file_name=xml_file,
    #                               plot_pianoroll=False,
    #                               plot_inline_ipynb=False,
    #                               save_at=None,
    #                               save_file_name=None,
    #                               do_save=False,
    #                               x_axis_res=2,
    #                               get_measure_onset=False)#, filter_dict=filter_dict_t)

    # part = m_df[['PartID']].to_numpy()
    # part_name = m_df[['Part Name']].to_numpy()
    # print(part)
    # print(np.unique(part, return_counts=True))
    # print(np.unique(part_name, return_counts=True))
    # print(len(part))


    # dur_pc_hist = mp.analyse.quarterlength_duration_histogram(m_df,
    #                                                           with_pitch=True,
    #                                                           do_plot=True)



