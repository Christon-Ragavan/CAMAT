"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""

try:
    from .parser_utils import *
    from .parser_utils import _get_file_path
    from .plot import pianoroll_parts
    from .xml_parser import XMLParser
    from .analyse import *
    from .analyse import _cs_total_parts,_cs_total_meas, _cs_pitch_histogram, _cs_initialize_df

except:
    from parser_utils import *
    from parser_utils import _get_file_path
    from plot import pianoroll_parts
    from xml_parser import XMLParser
    from analyse import *
    from analyse import _cs_total_parts,_cs_total_meas, _cs_pitch_histogram,_cs_initialize_df


import os
from os.path import basename, isdir, join

import pandas as pd

logger = set_up_logger(__name__)


@pianoroll_parts
def with_xml_file(file: str,
                  plot_pianoroll: bool = False,
                  plot_inline_ipynb: bool=False,
                  save_at: str = None,
                  save_file_name: str = None,
                  do_save: bool = False,
                  x_axis_res=2,
                  get_measure_onset:bool=False, get_upbeat_info=False,
                  filter_dict=None, *args, **kwargs) -> tuple[pd.DataFrame, bool, bool, list, int, bool,tuple[pd.DataFrame]]:

    file = _get_file_path(file=file)

    if '.xml' not in basename(file):
        e = "Not a .XML file, Only .xml file is supported. Use Musescore or finale to convert to .xml format"
        logger.error(e)
        raise Exception(e)
    save_at_fn = ''

    if do_save:
        if save_file_name is not None:
            if '.csv' not in basename(save_file_name):
                e = f"{save_file_name} format not supported. File name must end with '.csv'. E.g. file_name.csv"
                logger.error(e)
                raise Exception(e)
        if save_at is None:
            save_at = join(os.getcwd().replace(basename(os.getcwd()), 'data'), 'exports')
            if isdir(save_at) == False: os.makedirs(save_at)
            if save_file_name is None:
                save_file_name = basename(file).replace('.xml', '.csv')
            save_at_fn = join(save_at, save_file_name)
        else:
            if save_file_name is None:
                save_file_name = basename(file).replace('.xml', '.csv')
            save_at_fn = join(save_at, save_file_name)

    logger.info("Extracting")
    parser_o = XMLParser(path=file, logger=logger)
    df_xml = parser_o.xml_parse()
    if get_upbeat_info:
        upbeat_info = parser_o.upbeat_measure_info
        return df_xml, plot_pianoroll, parser_o.measure_onset_list, upbeat_info, x_axis_res, get_measure_onset, measure_onset_data, plot_inline_ipynb

    if filter_dict is not None:
        df_xml = filter(df_xml, filter_dict)

    measure_onset_data = parser_o._compute_measure_n_onset()
    if do_save:
        df_xml.to_csv(save_at_fn, sep=';')
    logger.info("Successful")
    return df_xml, plot_pianoroll, parser_o.measure_onset_list, x_axis_res, get_measure_onset,measure_onset_data, plot_inline_ipynb


def corpus_study(xml_files):
    df_list = []
    if 'https' in xml_files[0]:
        print("Please download the files and save it in the data folder")
        raise Exception("Please download the files and save it in the data folder")

    FileNames = [i.replace('.xml', '') for i in xml_files]
    df_data = _cs_initialize_df(FileNames)
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
    df_data = _cs_total_parts(df_data, df_list)
    df_data = _cs_total_meas(df_data, df_list)
    df_data = _cs_pitch_histogram(df_data, df_list, FileNames)
    # df_data = _cs_pitchclass_histogram(df_data, df_list, FileNames)

    print(df_data)

def testing():
    # xml_file = 'BrumAn_Bru1011_COM_3-6_MissaProde_002_01134.xml'
    # xml_file = 'MahGu_IGM11_COM_1-5_SymphonyNo_001_00334.xml'
    # xml_file = 'BuDi_Op1_2-7_COM_TrioSonata_001_00066.xml'
    # xml_file = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'
    # xml_file = 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml'
    xml_file = 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml'
    xml_file = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'
    xml_file = 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml'

    # xml_file = 'ElEd_Op39_1-5_COM_PompandCir_001_00292.xml'
    # xml_file = 'weird2.xml'
    # xml_file = 'weird33.xml'
    # xml_file = 'ultimate_tie_test.xml'

    # filter_dict_t = {'Measure': '2-5', 'PartID': '1-4'}



    d = with_xml_file(file=xml_file,
                      plot_pianoroll=True,
                      save_at=None,
                      save_file_name=None,
                      do_save=False, get_upbeat_info=False,#filter_dict=filter_dict_t,
                      x_axis_res=1)
    # print(d)
    # import analyse
    # analyse.ambitus(d)


if __name__ == "__main__":
    testing()

