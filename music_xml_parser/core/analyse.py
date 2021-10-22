import os
import re
import sys

sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))
import numpy as np

np.seterr(all="ignore")
try:
    from utils import midi2str, midi2pitchclass, pitchclassid2pitchclass
    from parser_utils import *
    from plot import *
except:
    from .utils import midi2str, midi2pitchclass, pitchclassid2pitchclass
    from .plot import *


def getVoice(df_data: pd.DataFrame):
    v = df_data['Voice']
    return list(set(v))

def search_upeat_files():
    pass


def get_part_info(df):
    v = df[['PartID', 'PartName']].drop_duplicates().to_numpy()
    return v


def max_measure_num(df_data: pd.DataFrame, part='all'):
    df_c = df_data.copy()
    df_c.drop_duplicates(subset='PartID', keep="last", inplace=True)
    df_c_n = df_c[['Measure', 'PartID', 'PartName']].to_numpy()

    return df_c_n


def metric_profile_split_time_signature(df_data: pd.DataFrame,
                                        plot_with=None,
                                        do_plot=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    ts = df_data['TimeSignature'].to_numpy()
    u, c = np.unique(ts, return_counts=True)
    mp_tc_dict = {}

    for ts_c in u:
        c_d = df_data.loc[df_data['TimeSignature'] == ts_c].copy()
        curr_h = metric_profile(c_d, x_label=f"Metric Profile (TimeSignature : {ts_c})",
                                plot_with=plot_with,
                                do_plot=do_plot)
        mp_tc_dict[ts_c] = curr_h
    return mp_tc_dict


def duration_histogram(df_data: pd.DataFrame,
                       with_pitch=False,
                       do_plot=True):
    pass


def time_signature_histogram(df_data: pd.DataFrame, do_plot=False, do_adjusted=False, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    if not do_adjusted:
        ts = df_data['TimeSignature'].to_numpy()
        ts_m = df_data[['TimeSignature', 'Measure']].drop_duplicates().to_numpy()
        ts_m = ts_m[:, 0]

        xlab = 'TimeSignature'
    else:
        xlab = 'TimeSignatureAdjusted'
        ts = df_data['TimeSignatureAdjusted'].to_numpy()
        ts_m = df_data[['TimeSignatureAdjusted', 'Measure']].drop_duplicates().to_numpy()
        ts_m = ts_m[:, 0]

    u, c = np.unique(ts_m, return_counts=True)
    if do_plot:
        barplot(u, counts=c, figsize='fit', x_label=xlab, y_label='Occurrences')
    data = [[i, int(c)] for i, c in zip(u, c)]
    data.sort(key=lambda x: x[1])

    return data


def ambitus(df_data: pd.DataFrame, output_as_midi=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)

    ab = []
    df_data['PartID'] = df_data['PartID'].astype(str)

    uni_parts = np.unique(df_data['PartID'].values)

    for i in uni_parts:

        d = df_data[df_data['PartID'].str.contains(i)].copy()

        d.dropna(subset=["MIDI"], inplace=True)
        max_r = np.max(d['MIDI'].to_numpy(dtype=float))
        min_r = np.min(d['MIDI'].to_numpy(dtype=float))
        diff_r = max_r - min_r
        name = np.unique(d['PartName'].to_numpy())[0]

        if output_as_midi:
            ab.append([int(i), str(name), int(min_r), int(max_r), int(diff_r)])
        else:

            min_r = midi2str(int(min_r))
            max_r = midi2str(int(max_r))
            ab.append([int(i),str(name), str(min_r), str(max_r), int(diff_r)])

    return ab


def pitch_histogram(df_data: pd.DataFrame,
                    do_plot=True,
                    do_plot_full_axis=True,
                    visulize_midi_range=None,
                    filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    df_c = df_data.copy()
    df_c.dropna(subset=["MIDI"], inplace=True)
    midi = df_c[['MIDI']].to_numpy()
    u, c = np.unique(midi, return_counts=True)
    if do_plot:
        barplot_pitch_histogram(u,
                                c,
                                do_plot_full_axis,
                                visulize_midi_range=visulize_midi_range)
    pitch_str = [midi2str(int(i)) for i in u]
    data = [[int(i), str(p), int(c)] for i, p, c in zip(u, pitch_str, c)]

    return data


def pitch_class_histogram(df_data: pd.DataFrame, x_axis_12pc=True, do_plot=True, filter_dict=None):
    if filter_dict is not None:
        df_data = filter(df_data, filter_dict)
    d = df_data.copy()

    d.dropna(subset=["Pitch"], inplace=True)
    d.dropna(subset=["MIDI"], inplace=True)
    d.drop(index=d[d['Pitch'] == 'rest'].index, inplace=True)

    p_o = d[['MIDI']].to_numpy()
    pc_int = [midi2pitchclass(int(i))[1] for i in p_o]

    u, c = np.unique(pc_int, return_counts=True)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    u_labels = [names[i] for i in u]

    if do_plot:
        barplot_pitch_class_histogram(u, c, u_labels, x_axis_12pc=x_axis_12pc)

    data = [[int(id), str(i), int(c)] for id, i, c in zip(u, u_labels, c)]
    data.sort(key=lambda x: x[0])
    data = [[i[1], i[2]] for i in data]
    return data


def quarterlength_duration_histogram(df_data: pd.DataFrame,
                                     plot_with=None,
                                     do_plot=True, filter_dict=None):
    df_c = df_data.copy()
    if plot_with == 'pitch':
        plot_with = 'Pitch'
    if plot_with == 'pitchclass':
        plot_with = 'PitchClass'
    if plot_with == 'none':
        plot_with = None

    if filter_dict is not None:
        df_c = filter(df_c, filter_dict)

    if plot_with == None:
        dur = df_c['Duration'].to_numpy(dtype=float)
        u, c = np.unique(dur, return_counts=True)
        # a.sort(key=lambda x: x[1])
        labels = u
        if do_plot:
            barplot_quaterlength_duration_histogram(labels, counts=c)
        data = [[round(float(i), 2), int(c)] for i, c in zip(u, c)]
        return data
    else:
        if plot_with == 'PitchClass':
            # print(df_c[['Pitch', 'Duration']])
            df_c.dropna(subset=["MIDI"], inplace=True)

            n_df = df_c[['MIDI', 'Duration']].to_numpy()

            p = [midi2pitchclass(i)[1] for i in n_df[:, 0]]

            # p = [midi2pitchclass(i)[0] for i in n_df[:, 0]]

            d = n_df[:, 1]
            n_df = [[i, ii] for i, ii in zip(p,d)]
            n_df = np.array(n_df,dtype='<U21')
            u, c = np.unique(n_df, axis=0, return_counts=True)
            p = [str(i) for i in u[:, 0]]
            p_str = [pitchclassid2pitchclass(int(i)) for i in u[:, 0]]

            d = [float(i) for i in u[:, 1]]
            pd_data_s = pd.DataFrame(np.array([p, d, c]).T, columns=['Pitch', 'Duration', 'Count'])
            convert_dict = {'Count': int, 'Duration': float}
            pd_data_s = pd_data_s.astype(convert_dict)
            data = pd_data_s.to_numpy()


            if do_plot:
                plot_3d_ql_pc(np.array(data),  ylabel='Quarter Length Duration')

            pd_data_s2 = pd.DataFrame(np.array([p_str, d, c]).T, columns=['Pitch', 'Duration', 'Count'])
            convert_dict2 = {'Count': int, 'Duration': float}
            pd_data_s2 = pd_data_s2.astype(convert_dict2)
            data2 = pd_data_s2.to_numpy()
            return data2

        elif plot_with == 'Pitch':

            df_c.dropna(subset=["MIDI"], inplace=True)

            n_df = df_c[['MIDI', 'Duration']].to_numpy(dtype=float)

            u, c = np.unique(n_df, axis=0, return_counts=True)
            p_str = [midi2str(int(i)) for i in u[:, 0]]
            p = [int(i) for i in u[:, 0]]
            d = [float(i) for i in u[:, 1]]
            pd_data_s = pd.DataFrame(np.array([p, d, c]).T, columns=['Pitch', 'Duration', 'Count'])
            pd_data_p_str = pd.DataFrame(np.array([p_str, d, c]).T, columns=['Pitch', 'Duration', 'Count'])
            convert_dict = {'Count': int,
                            'Duration': float
                            }
            pd_data_s = pd_data_s.astype(convert_dict)
            data = pd_data_s.to_numpy()

            pd_data_p_str = pd_data_p_str.astype(convert_dict)
            pd_data_p_str = pd_data_p_str.to_numpy()

            if do_plot:
                plot_3d(np.array(data), ylabel='Quarter Length Duration')
            return pd_data_p_str
        else:
            print("Please either enter PitchClass or Pitch or None ")



def interval(df_data: pd.DataFrame, part=None, do_plot=True, filter_dict=None):
    # v = df_data[['PartID', 'PartName']].drop_duplicates().to_numpy()
    df_c = df_data.copy()
    if filter_dict is not None:
        df_c = filter(df_c, filter_dict)
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
                   plot_with=None,
                   do_plot=True,
                   filter_dict=None):
    df_c = df_data.copy()
    if filter_dict is not None:
        df_c = filter(df_c, filter_dict)

    df_c.dropna(subset=["MIDI"], inplace=True)
    df_c['metricprofile'] = pd.to_numeric(df_c['Onset']) - pd.to_numeric(df_c['MeasureOnset'])
    if plot_with == None:
        u, c = np.unique(df_c['metricprofile'].to_numpy(dtype=float), axis=0, return_counts=True)
        u = [i + 1 for i in u]
        if do_plot:
            barplot_mp(u, counts=c, x_label=x_label, y_label='Occurrences')
        data = [[round(float(i),2), int(c)] for i, c in zip(u, c)]
        return data
    else:
        if plot_with == 'Pitch':

            n_df = df_c[['MIDI', 'metricprofile']].to_numpy(dtype=float)
            u, c = np.unique(n_df, axis=0, return_counts=True)
            p = [int(i) for i in u[:, 0]]
            pitch = [midi2str(int(i)) for i in u[:, 0]]

            pd_data_s = pd.DataFrame(np.array([p, u[:, 1], c]).T, columns=['Pitch', 'metricprofile', 'Count'])
            convert_dict = {'Count': int, 'metricprofile': float}
            pd_data_s = pd_data_s.astype(convert_dict)
            data = pd_data_s.to_numpy()

            if do_plot:
                beat_stength_3d(data, ylabel=x_label, plot_with=plot_with)
            data_f = pd.DataFrame(np.array([p, pitch, u[:, 1], c]).T, columns=['MIDI', 'Pitch', 'metricprofile', 'Count'])
            convert_dict_2 = {'Count': int, 'metricprofile': float}
            data_f = data_f.astype(convert_dict_2)
            data_2 = data_f.to_numpy()

            return data_2
        elif plot_with == 'PitchClass':

            n_df = df_c[['MIDI', 'metricprofile']].to_numpy(dtype=float)
            m_pc = np.array([midi2pitchclass(i)[1] for i in n_df[:,0]])

            n_df[:,0] = m_pc
            # for i u in zip(n_df:
            #     print(u)
            # print(n_df)
            u, c = np.unique(n_df, axis=0, return_counts=True)

            p = [int(i) for i in u[:, 0]]

            pc_n = [pitchclassid2pitchclass(int(i)) for i in u[:, 0]]

            pd_data_s = pd.DataFrame(np.array([p, u[:, 1], c]).T, columns=['PitchClass', 'metricprofile', 'Count'])
            convert_dict = {'Count': int, 'metricprofile': float}
            pd_data_s = pd_data_s.astype(convert_dict)
            data = pd_data_s.to_numpy()

            if do_plot:
                beat_stength_3d(data, ylabel= x_label, plot_with=plot_with)
            data_f = pd.DataFrame(np.array([pc_n, u[:, 1], c]).T,
                                  columns=[ 'Pitch', 'metricprofile', 'Count'])
            convert_dict_2 = {'Count': int, 'metricprofile': float}
            data_f = data_f.astype(convert_dict_2)
            data_2 = data_f.to_numpy()

            return data_2



def filter(df_data: pd.DataFrame, filter_dict):
    """
    Order of the dict is important
    :param df_data:
    :param filter_dict:
    :return:
    """
    f_df = df_data.copy()
    for i in filter_dict:
        s_d = str(filter_dict[i])
        if '-' in s_d:
            s, e = re.split('-', s_d, 2)
            arr = np.arange(int(s), int(e) + 1)
            df_list = []
            for ii in arr:
                grouped = f_df.groupby(by=[i])
                try:
                    df_list.append(grouped.get_group(str(ii)).copy())
                except:
                    df_list.append(grouped.get_group(int(ii)).copy())
            f_df = pd.concat(df_list,
                             ignore_index=True,
                             verify_integrity=False,
                             copy=False)
        else:
            grouped = f_df.groupby(by=[i])
            try:
                s_d = int(s_d)
                f_df = grouped.get_group(s_d).copy()
            except:
                s_d = str(s_d)
                f_df = grouped.get_group(s_d).copy()
    return f_df


def _cs_initialize_df(df_row_name):

    c = ['FileName', 'TotalMeasure', 'PitchClass']
    df_data = pd.DataFrame(df_row_name, columns=c)
    return df_data


def _cs_total_parts(df_data, dfs):
    t_parts_list = []
    for df in dfs:
        df_p = len(list(set(df['PartID'].drop_duplicates().to_numpy())))
        t_parts_list.append(df_p)
    df_data['TotalParts'] = t_parts_list
    return df_data


def _cs_total_meas(df_data, dfs):
    t_meas_list = []
    for df in dfs:
        df_p = len(list(set(df['Measure'].drop_duplicates().to_numpy())))
        t_meas_list.append(df_p)
    df_data['TotalMeasure'] = t_meas_list
    return df_data


def _cs_ambitus(df_data, dfs, FileNames):
    ph_data = []
    for df in dfs:
        data = ambitus(df,
                       output_as_midi=True,
                       filter_dict=None)
        data = np.array(data)
        ph_data.append(data)

    # for idx, p in enumerate(ph_data):
    # data_f = np.full(shape=(12, 2), fill_value='')
    # data_f[:, 0] = pc_names
    # data_f = pd.DataFrame(data_f, columns=['PitchCLass', 'Freq'])
    # for i in p:
    #     r = data_f.index[data_f['PitchCLass'] == str(i[0])]
    #     data_f.loc[r, 'Freq'] = str(i[1])
    # data_f = data_f.T
    # n_l = [FileNames[idx] for _ in range(len(data_f))]
    # col_names =  ['PitcClasshHist' for _ in range (len(data_f.columns))]
    # data_f.columns = col_names
    #
    # data_f['FileName'] = n_l
    #
    # data_all.append(data_f)
    # del data_f
    return df_data


def _cs_pitch_histogram(df_data, dfs, FileNames):
    midi_min_max = []
    ph_data = []
    for df in dfs:
        data = pitch_histogram(df,
                               do_plot=False,
                               do_plot_full_axis=False,
                               visulize_midi_range=None,
                               filter_dict=None)
        data = np.array(data)
        midi_min_max.append(min(data[:, 0]))
        midi_min_max.append(max(data[:, 0]))
        ph_data.append(data)

    axis_range = np.arange(start=int(min(midi_min_max)), stop=int(max(midi_min_max)))
    data_all = []
    for idx, p in enumerate(ph_data):
        data_f = np.full(shape=(len(axis_range), 3), fill_value=0)
        data_f[:, 0] = axis_range
        data_f = pd.DataFrame(data_f, columns=['Midi', 'Pitch', 'Freq'])

        for i in p:
            r = data_f.index[data_f['Midi'] == int(i[0])]
            data_f.loc[r, 'Pitch'] = str(i[1])
            data_f.loc[r, 'Freq'] = int(i[2])
        data_f = data_f.T
        n_l = [FileNames[idx] for _ in range(len(data_f))]
        col_names = ['PitchHist' for _ in range(len(data_f.columns))]
        data_f.columns = col_names

        data_f['FileName'] = n_l

        data_all.append(data_f)
        del data_f
    data_all = pd.concat(data_all, ignore_index=True, sort=False)

    df_data = df_data.merge(data_all, on='FileName', how='outer')  # ,suffixes = ('', '_n'))

    return df_data


def _cs_pitchclass_histogram(df_data, dfs, FileNames):
    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    midi_min_max = []
    ph_data = []
    for idx, df in enumerate(dfs):
        data = pitch_class_histogram(df,
                                     do_plot=False,
                                     x_axis_12pc=True,
                                     filter_dict=None)
        ph_data.append(data)
    data_all = []
    for idx, p in enumerate(ph_data):
        data_f = np.full(shape=(12, 2), fill_value='')
        data_f[:, 0] = pc_names
        data_f = pd.DataFrame(data_f, columns=['PitchCLass', 'Freq'])
        for i in p:
            r = data_f.index[data_f['PitchCLass'] == str(i[0])]
            data_f.loc[r, 'Freq'] = str(i[1])
        data_f = data_f.T
        n_l = [FileNames[idx] for _ in range(len(data_f))]
        col_names = ['PitcClasshHist' for _ in range(len(data_f.columns))]
        data_f.columns = col_names
        data_f['FileName'] = n_l
        data_all.append(data_f)
        del data_f
    # col_names.append('FileName')
    data_all = pd.concat(data_all, ignore_index=True, sort=False)
    # data_all = data_all.drop_duplicates(subset=col_names, keep='first')
    # df_data = df_data.merge(data_all, left_index=True, right_index=True, how='outer')
    df_data = df_data.merge(data_all, left_index=True, right_index=True, how='outer')  # ,suffixes = ('', '_n'))
    # df_data['FileName_x'] = df_data['FileName_y'].tolist()
    df_data = df_data.rename(columns={'FileName_x': 'FileName'})
    df_data.drop(df_data.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    # # print(data_all)
    return df_data


# def testing():
#
#     xml_files = ['PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml', 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml']
#
#     out = corpus_study(xml_files)
#
#
# if __name__=='__main__':
#     testing()
