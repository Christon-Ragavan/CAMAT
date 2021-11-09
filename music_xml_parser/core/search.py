import numpy as np
import sys
import os
import math

sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))

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

def simple_interval_search(xml_file, interval):
    i_len = len(interval)
    c_df = with_xml_file(file=xml_file,
                         plot_pianoroll=False,
                         plot_inline_ipynb=False,
                         save_at=None,
                         save_file_name=None,
                         do_save=False,
                         get_measure_Onset=False)
    i_df = _compute_intervals(df_data=c_df)
    s_df = i_df[["Onset", "Duration", "Pitch", "Octave", "MIDI", "Measure", "PartID", "PartName", "Interval"]].copy()

    int_l = i_df['Interval'].tolist()
    sel_index = []

    for s in range(len(int_l)):
        e = s+i_len
        tc = int_l[s:e]
        if interval == tc:
            sel_index.append(s)
    sel_dfs = []
    for si in sel_index:
        s_df_t = s_df.iloc[si:si+i_len].copy()
        sel_dfs.append(s_df_t)
    if len(sel_dfs)==0:
        print("No search found")
    return sel_dfs

if __name__ == '__main__':
    xml_files = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'


    df = simple_interval_search(xml_files, interval=[2, 2])
    for i in df:
        print(i)