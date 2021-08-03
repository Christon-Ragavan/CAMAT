from os.path import isdir, isfile, basename, join
import pathlib
import traceback
import logging
import numpy as np
import pandas as pd
try:
    from .parser_utils import *
    from .parser_utils import _get_file_path
    from .plot import pianoroll_parts
    from .xml_parser import XMLParser
except:
    from parser_utils import *
    from parser_utils import _get_file_path
    from plot import pianoroll_parts
    from xml_parser import XMLParser

import os
from os.path import isdir, isfile, basename, join
import sys

logger = set_up_logger(__name__)


@pianoroll_parts
def with_xml_file(file_name: str, plot_pianoroll: bool = False, save_at: str = None,
                  save_file_name: str = None, do_save: bool = False, *args, **kwargs) -> tuple[
    pd.DataFrame, bool, list]:
    file = _get_file_path(file_name=file_name)

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
    if do_save:
        df_xml.to_csv(save_at_fn, sep=';')
    logger.info("Successful")
    print(df_xml)
    print(parser_o.measure_offset_list)
    return df_xml, plot_pianoroll, parser_o.measure_offset_list


if __name__ == "__main__":
    xml_file = 'BrumAn_Bru1011_COM_3-6_MissaProde_002_01134.xml'

    d = with_xml_file(file_name=xml_file, plot_pianoroll=True,
                      save_at=None, save_file_name=None,
                      do_save=False)
    print("------")
