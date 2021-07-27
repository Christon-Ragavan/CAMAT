import sys
from os.path import isdir, isfile, basename, join
import pathlib
import traceback
import logging
import numpy as np

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

logger.debug(np.random.rand(2))