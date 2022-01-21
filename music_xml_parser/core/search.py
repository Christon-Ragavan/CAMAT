import numpy as np
import sys
import os
import math
import pandas as pd
sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))
sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'core'), ''))

try:
    from .analyse import interval
    from .parse import with_xml_file
except:
    from analyse import interval
    from parse import with_xml_file

np.seterr(all="ignore")

def _compute_intervals(df_data):

    p_df1 = df_data.copy()
    u_parts = np.unique(df_data['PartID'].to_numpy())
    u_parts = [int(i) for i in u_parts]
    intervals_p = []
    df_idxs_p = []
    part_len = []
    for c_p in u_parts:
        grouped = p_df1.groupby(p_df1.PartID)
        try:
            p_all = grouped.get_group(int(c_p)).copy()
        except:
            p_all = grouped.get_group(str(c_p)).copy()
        pd_idxs = p_all.index.values.tolist()
        df_idxs_p.extend(pd_idxs)
        midi_p = p_all['MIDI'].to_numpy()
        c_intvl = [np.nan, ]
        for s, t in zip(midi_p, midi_p[1:]):
            if math.isnan(t) or math.isnan(s):
                a = np.nan
            else:
                a = int(t-s)
            c_intvl.append(a)
        part_len.append(len(c_intvl))
        intervals_p.extend(c_intvl)

    assert len(df_data) == np.sum(part_len), f"Intervals and Df Data length not matching"
    assert len(intervals_p) == len(df_idxs_p), f"interval and df_idx are not matching"
    for idx, i in zip(df_idxs_p, intervals_p):
        df_data.at[idx, 'Interval'] = i
    return df_data

def simple_interval_search(xml_file, interval, return_details=False):
    i_len = len(interval)
    c_df = with_xml_file(file=xml_file,
                         plot_pianoroll=False,
                         plot_inline_ipynb=False,
                         save_at=None,
                         save_file_name=None,
                         do_save=False,
                         get_measure_Onset=False)
    i_df = _compute_intervals(df_data=c_df)
    s_df = i_df[["Pitch", "Octave", "MIDI",  "PartName", "PartID", "Measure", "LocalOnset","Onset"]].copy()
    int_l = i_df['Interval'].tolist()
    sel_index = []

    for s in range(len(int_l)):
        e = s+i_len
        tc = int_l[s:e]
        if interval == tc: # checking exact interval match
            sel_index.append(s-1)

    if len(sel_index)==0:
        print("No Interval Found")
        return 'No Interval Found'

    else:
        sel_dfs = []
        sel_pitchs = []
        for si in sel_index:
            s_df_t_1 = s_df.iloc[si].copy()
            sel_dfs.append(s_df_t_1.tolist())
            sel_pitchs.append(s_df_t_1['Pitch'])
        p_c = np.array(np.unique(sel_pitchs, return_counts=True)).T
        p_c_s = p_c[p_c[:, 1].argsort()[::-1]]
        sel_dfs = np.array(sel_dfs)
        p_c_s = np.array(p_c_s)

        sel_dfs = pd.DataFrame(list(zip(sel_dfs[:, 0], sel_dfs[:, 1],sel_dfs[:, 2],sel_dfs[:, 3],sel_dfs[:, 4],sel_dfs[:, 5], sel_dfs[:, 6], sel_dfs[:, 7])), columns=["Pitch", "Octave", "MIDI",  "PartName", "PartID", "Measure","LocalOnset","Onset"])
        p_c_s = pd.DataFrame(list(zip(p_c_s[:, 0], p_c_s[:, 1])), columns=["Pitch", "Occurance"])

        if return_details:
            return sel_dfs
        else:
            return p_c_s

if __name__=="__main__":
    xml_file = 'https://analyse.hfm-weimar.de/database/03/MoWo_K171_COM_1-4_StringQuar_003_00867.xml'

    results_01 = simple_interval_search(xml_file,
                                       interval=[2, 2],
                                       return_details=False)

    print(results_01)  # durch diesen Befehl wird eine einfach Tabelle dargestellt.