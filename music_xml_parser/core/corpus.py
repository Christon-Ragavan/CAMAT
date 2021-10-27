import numpy as np
import sys
import os
#sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'core'), ''))
sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))

try:
    from .analyse import _cs_get_part_list, _cs_initialize_df, _cs_total_parts, _cs_total_meas, _cs_ambitus, _cs_time_signature, _cs_pitchclass_histogram, _cs_interval
    from .parse import with_xml_file
except:
    from analyse import _cs_get_part_list, _cs_initialize_df, _cs_total_parts, _cs_total_meas, _cs_ambitus, _cs_time_signature, _cs_pitchclass_histogram, _cs_interval
    from parse import with_xml_file


np.seterr(all="ignore")


def analyse_basic_statistics(xml_files, get_in_midi=False):
    """
    # Basic Statistics - Ambitues, total measure, parts, Timesignatures (in one cell)
    # Pitchclass
    # Intervals +/-12 notes (and <> end bins)

    :param xml_files:
    :return:
    """
    df_list = []
    if 'https' in xml_files[0]:
        print("Please download the files and save it in the data folder")
        raise Exception("Please download the files and save it in the data folder")

    FileNames = [i.replace('.xml', '') for i in xml_files]


    for xf in xml_files:
        c_df = with_xml_file(file=xf,
                                   plot_pianoroll=False,
                                   plot_inline_ipynb=False,
                                   save_at=None,
                                   save_file_name=None,
                                   do_save=False,
                                   x_axis_res=2,
                                   get_measure_Onset=False)
        df_list.append(c_df)
    part_info = _cs_get_part_list(df_list)
    df_data = _cs_initialize_df(FileNames, part_info)
    df_data = _cs_total_parts(df_data, df_list, part_info)
    df_data = _cs_total_meas(df_data, df_list, part_info)
    df_data = _cs_ambitus(df_data, df_list, get_in_midi=get_in_midi)
    df_data = _cs_time_signature(df_data, df_list,part_info)
    # df_data = _cs_pitchclass_histogram(df_data, df_list, part_info)
    return df_data

def analyse_pitch_class(xml_files, include_basic_stats=True, get_in_midi=False):
    """
    # Basic Statistics - Ambitues, total measure, parts, Timesignatures (in one cell)
    # Pitchclass
    # Intervals +/-12 notes (and <> end bins)

    :param xml_files:
    :return:
    """
    df_list = []
    if 'https' in xml_files[0]:
        print("Please download the files and save it in the data folder")
        raise Exception("Please download the files and save it in the data folder")

    FileNames = [i.replace('.xml', '') for i in xml_files]


    for xf in xml_files:
        c_df = with_xml_file(file=xf,
                                   plot_pianoroll=False,
                                   plot_inline_ipynb=False,
                                   save_at=None,
                                   save_file_name=None,
                                   do_save=False,
                                   x_axis_res=2,
                                   get_measure_Onset=False)
        df_list.append(c_df)
    part_info = _cs_get_part_list(df_list)
    df_data = _cs_initialize_df(FileNames, part_info)

    if include_basic_stats:
        df_data = _cs_total_parts(df_data, df_list, part_info)
        df_data = _cs_total_meas(df_data, df_list, part_info)
        df_data = _cs_ambitus(df_data, df_list, get_in_midi=get_in_midi)
        df_data = _cs_time_signature(df_data, df_list,part_info)
    df_data = _cs_pitchclass_histogram(df_data, df_list, part_info)
    return df_data

def analyse_interval(xml_files, include_basic_stats=True, get_in_midi=False):
    """
    # Basic Statistics - Ambitues, total measure, parts, Timesignatures (in one cell)
    # Pitchclass
    # Intervals +/-12 notes (and <> end bins)

    :param xml_files:
    :return:
    """
    df_list = []
    if 'https' in xml_files[0]:
        print("Please download the files and save it in the data folder")
        raise Exception("Please download the files and save it in the data folder")

    FileNames = [i.replace('.xml', '') for i in xml_files]


    for xf in xml_files:
        c_df = parse.with_xml_file(file=xf,
                                   plot_pianoroll=False,
                                   plot_inline_ipynb=False,
                                   save_at=None,
                                   save_file_name=None,
                                   do_save=False,
                                   x_axis_res=2,
                                   get_measure_Onset=False)
        df_list.append(c_df)
    part_info = _cs_get_part_list(df_list)
    df_data = _cs_initialize_df(FileNames, part_info)

    if include_basic_stats:
        df_data = _cs_total_parts(df_data, df_list, part_info)
        df_data = _cs_total_meas(df_data, df_list, part_info)
        df_data = _cs_ambitus(df_data, df_list, get_in_midi=get_in_midi)
        df_data = _cs_time_signature(df_data, df_list,part_info)
    df_data = _cs_pitchclass_histogram(df_data, df_list, part_info)
    df_data = _cs_interval(df_data, df_list, part_info)
    return df_data
if __name__ == '__main__':
    xml_files = ['PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml', 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml']
    df = analyse_pitch_class(xml_files, include_basic_stats=True)
