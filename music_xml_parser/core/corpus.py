import numpy as np
import analyse
import parse

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
        c_df = parse.with_xml_file(file=xf,
                                   plot_pianoroll=False,
                                   plot_inline_ipynb=False,
                                   save_at=None,
                                   save_file_name=None,
                                   do_save=False,
                                   x_axis_res=2,
                                   get_measure_Onset=False)
        df_list.append(c_df)
    part_info = analyse._cs_get_part_list(df_list)
    df_data = analyse._cs_initialize_df(FileNames, part_info)
    df_data = analyse._cs_total_parts(df_data, df_list, part_info)
    df_data = analyse._cs_total_meas(df_data, df_list, part_info)
    df_data = analyse._cs_ambitus(df_data, df_list, get_in_midi=get_in_midi)
    df_data = analyse._cs_time_signature(df_data, df_list,part_info)
    # df_data = analyse._cs_pitchclass_histogram(df_data, df_list, part_info)
    print(df_data)

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
        c_df = parse.with_xml_file(file=xf,
                                   plot_pianoroll=False,
                                   plot_inline_ipynb=False,
                                   save_at=None,
                                   save_file_name=None,
                                   do_save=False,
                                   x_axis_res=2,
                                   get_measure_Onset=False)
        df_list.append(c_df)
    part_info = analyse._cs_get_part_list(df_list)
    df_data = analyse._cs_initialize_df(FileNames, part_info)

    if include_basic_stats:
        df_data = analyse._cs_total_parts(df_data, df_list, part_info)
        df_data = analyse._cs_total_meas(df_data, df_list, part_info)
        df_data = analyse._cs_ambitus(df_data, df_list, get_in_midi=get_in_midi)
        df_data = analyse._cs_time_signature(df_data, df_list,part_info)
    df_data = analyse._cs_pitchclass_histogram(df_data, df_list, part_info)
    print(df_data)

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
    part_info = analyse._cs_get_part_list(df_list)
    df_data = analyse._cs_initialize_df(FileNames, part_info)

    if include_basic_stats:
        df_data = analyse._cs_total_parts(df_data, df_list, part_info)
        df_data = analyse._cs_total_meas(df_data, df_list, part_info)
        df_data = analyse._cs_ambitus(df_data, df_list, get_in_midi=get_in_midi)
        df_data = analyse._cs_time_signature(df_data, df_list,part_info)
    df_data = analyse._cs_pitchclass_histogram(df_data, df_list, part_info)
    df_data = analyse._cs_interval(df_data, df_list, part_info)
    print(df_data)

def testing():
    xml_file = 'https://analyse.hfm-weimar.de/database/03/MoWo_K171_COM_1-4_StringQuar_003_00867.xml'

    m_df = parse.with_xml_file(file=xml_file,
                               plot_pianoroll=False,
                               save_at=None,
                               save_file_name=None,
                               do_save=False, get_upbeat_info=False,
                               x_axis_res=1)
    out = analyse.ambitus(m_df, output_as_midi=True)
    print(out)


if __name__ == '__main__':
    xml_files = ['PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml', 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml']
    analyse_pitch_class(xml_files, include_basic_stats=True)
