"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""

try:
    from .parser_utils import *
    from .parser_utils import _get_file_path
    from .plot import pianoroll_parts
    from .xml_parser import XMLParser
    from .analyse import filter
except:
    from parser_utils import *
    from parser_utils import _get_file_path
    from plot import pianoroll_parts
    from xml_parser import XMLParser
    from analyse import filter


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
                  get_measure_onset:bool=False,
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

    if filter_dict is not None:
        df_xml = filter(df_xml, filter_dict)

    measure_offset_data = parser_o._compute_measure_n_offset()
    if do_save:
        df_xml.to_csv(save_at_fn, sep=';')
    logger.info("Successful")
    return df_xml, plot_pianoroll, parser_o.measure_offset_list, x_axis_res, get_measure_onset,measure_offset_data, plot_inline_ipynb

def testing():
    # xml_file = 'BrumAn_Bru1011_COM_3-6_MissaProde_002_01134.xml'
    # xml_file = 'MahGu_IGM11_COM_1-5_SymphonyNo_001_00334.xml'
    # xml_file = 'BuDi_Op1_2-7_COM_TrioSonata_001_00066.xml'
    # xml_file = 'PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml'
    xml_file = 'BaJoSe_BWV18_COM_5-5_CantataGle_004_00110.xml'

    # filter_dict_t = {'Measure': '2-5', 'PartID': '1-4'}

    d = with_xml_file(file=xml_file,
                      plot_pianoroll=False,
                      save_at=None,
                      save_file_name=None,
                      do_save=False,#filter_dict=filter_dict_t,
                      x_axis_res=1)
    print(d)
    # import analyse
    # analyse.ambitus(d)

if __name__ == "__main__":
    testing()

