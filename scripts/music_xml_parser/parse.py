import sys
from os.path import isdir, isfile, basename, join
import pathlib
import traceback
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s:%(module)s')
file_handler = logging.FileHandler('logs.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


logger.debug("Testing the logging function")
logger.info("Testing the logging function")
logger.debug(np.random.rand(20))