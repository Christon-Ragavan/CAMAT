import numpy as np
import sys
import os

sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))
import os
try:
    from .analyse import _cs_get_part_list, _cs_initialize_df, _cs_total_parts, _cs_total_meas, _cs_ambitus, \
        _cs_time_signature, _cs_pitchclass_histogram, _cs_interval
    from .parse import with_xml_file
    from .web_scrapper import get_file_from_server
except:
    from analyse import _cs_get_part_list, _cs_initialize_df, _cs_total_parts, _cs_total_meas, _cs_ambitus, \
        _cs_time_signature, _cs_pitchclass_histogram, _cs_interval
    from parse import with_xml_file
    from web_scrapper import get_file_from_server

np.seterr(all="ignore")


def analyse_basic_statistics(xml_files, get_in_midi=False):
    """
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
    df_data = _cs_time_signature(df_data, df_list, part_info)
    return df_data


def analyse_pitch_class(xml_files, include_basic_stats=True, get_in_midi=False):
    """
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
        df_data = _cs_time_signature(df_data, df_list, part_info)
    df_data = _cs_pitchclass_histogram(df_data, df_list, part_info)
    return df_data


def analyse_interval(xml_files, separate_parts=True,
                     interval_range=None,
                     include_basic_stats=True,
                     include_pitchclass=False,
                     get_full_axis=False,
                     get_in_midi=False,
                     get_in_percentage=False):
    """
    :param xml_files:
    :return:
    """
    if separate_parts == False:
        print("TOBE Implemented, currently setting separate_parts==True ")
        separate_parts = True
    # if max(interval_range) >12:
    #     print("interval_range > 12 to be implemented, instead using get_full_axis=True")
    #     get_full_axis=True
    # if min(interval_range)<-12:
    #     print("interval_range < -12 to be implemented, instead using get_full_axis=True")
    #     get_full_axis = True
    if interval_range is None:
        get_full_axis = True
    if not get_full_axis:
        if interval_range is None:
            get_full_axis = True
        else:
            assert len(interval_range) == 2, "Invalid interval range: Example interval_range=[-12, 12]"
    else:
        interval_range = None

    df_list = []
    if 'https' in xml_files[0]:
        n_xml_files = []
        for xf in xml_files:
            f = get_file_from_server(xf)
            n_xml_files.append(os.path.basename(f))
    else:
        n_xml_files = xml_files
        # raise Exception("Please download the files and save it in the data folder")

    FileNames = [i.replace('.xml', '') for i in n_xml_files]

    for xf in n_xml_files:
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
        df_data = _cs_time_signature(df_data, df_list, part_info)
    if include_pitchclass:
        df_data = _cs_pitchclass_histogram(df_data, df_list, part_info=part_info, separate_parts=separate_parts,
                                           get_in_percentage=get_in_percentage)
    df_data = _cs_interval(df_data,
                           df_list,
                           part_info=part_info,
                           separate_parts=separate_parts,
                           get_full_axis=get_full_axis,
                           interval_range=interval_range, get_in_percentage=get_in_percentage)
    return df_data


if __name__ == '__main__':
    # xml_files = ['PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml', 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml']
    xml_files = ['https://analyse.hfm-weimar.de/database/03/MoWo_K171_COM_1-4_StringQuar_003_00867.xml',
                'https://analyse.hfm-weimar.de/database/03/MoWo_K171_COM_2-4_StringQuar_003_00868.xml',
                'https://analyse.hfm-weimar.de/database/03/MoWo_K171_COM_3-4_StringQuar_003_00869.xml',
                'https://analyse.hfm-weimar.de/database/03/MoWo_K171_COM_4-4_StringQuar_003_00870.xml']


    df = analyse_interval(xml_files,
                          separate_parts=True,
                          # To get info on part level. Currently only True is working
                          interval_range=[-6, 6],
                          # Please give two number, first a lower number followed by greater -> example[-7, 7]
                          include_basic_stats=False,
                          include_pitchclass=False,
                          get_full_axis=False,
                          get_in_percentage=False)  # If true you will get full min and max interval axis of all the files in xml_files(list)

    print(df)