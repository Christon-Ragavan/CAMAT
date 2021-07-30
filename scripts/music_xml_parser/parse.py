import sys
from os.path import isdir, isfile, basename, join
import pathlib
import traceback
import logging
import numpy as np
import pandas as pd
from utils import *
from plot import plotting_wrapper_parts
from xml_parser import xml_parse
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

def with_xml_files(file: str, plot_pianoroll: bool, save_at: str = None, save_file_name: str = None, *args,
                   **kwargs) -> pd.DataFrame:
    if '.xml' not in basename(file):
        e = "Not a .XML file, Only .xml file is supported. Use Musescore or finale to convert to .xml format"
        logger.error(e)
        raise Exception(e)
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
    df_xml = xml_parse(path=file, logger=logger)
    df_xml.to_csv(save_at_fn, sep=';')
    logger.info("Successful")

    return df_xml


if __name__ == "__main__":
    xml_file = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files/ultimate_tie_test.xml'
    with_xml_files(file=xml_file, plot_pianoroll=True, save_at=None, save_file_name=None)

"""
search_config/file_link -> scrape_database from web -> download in local folder -> read from local folder -> xml to pandas -> save and return pandas df -> pianoroll
todo: jupyternote simple  - gie the link and I give piano roll

Pack in a folder + jupyter 
- Friday - setup scripts for students 
- Monday - testing  
"""
