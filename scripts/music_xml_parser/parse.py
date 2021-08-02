import sys
from os.path import isdir, isfile, basename, join
import pathlib
import traceback
import logging
import numpy as np
import pandas as pd
from utils import *
from plot import pianoroll_parts
from xml_parser import XMLParser
import os
from os.path import isdir, isfile, basename, join

logger = set_up_logger(__name__)

"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s')
file_handler = logging.FileHandler('logs.log', 'w')
file_handler.setFormatter(formatter)
stream_formatter = logging.Formatter('%(levelname)s:%(name)s:%(lineno)d:%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.info("Music xml Parser - score to pandas")

"""


# Decorator search
# Decorator scrape_database


@pianoroll_parts
def with_xml_file(file: str, plot_pianoroll: bool = False, save_at: str = None,
                  save_file_name: str = None, do_save: bool = False, *args, **kwargs) -> tuple[
    pd.DataFrame, bool, list]:
    if '\\' in file:
        file = file.replace('\\', '')
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
    return df_xml, plot_pianoroll, parser_o.measure_offset_list


if __name__ == "__main__":
    xml_file = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files/ultimate_tie_test.xml'
    # xml_file = 'C:/Users/egor_/Desktop/weimar/ultimate_tie_test.xml'

    d = with_xml_file(file=xml_file, plot_pianoroll=True, save_at=None, save_file_name=None, do_save=False)
    print("------")
    print(d)

"""
search_config/file_link -> scrape_database from web -> download in local folder -> read from local folder -> xml to pandas -> save and return pandas df -> pianoroll
todo: jupyternote simple  - gie the link and I give piano roll

Pack in a folder + jupyter 
- Friday - setup scripts for students 
- Monday - testing  
"""
