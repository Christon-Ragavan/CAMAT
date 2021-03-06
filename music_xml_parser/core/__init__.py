#!/usr/bin/python

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2021 www.hfm-weimar.de  Hochschule für Musik FRANZ LISZT Weimar

__author__ = "Christon Nadar"
__copyright__ = "Copyright (c) 2021 www.hfm-weimar.de  Hochschule für Musik FRANZ LISZT Weimar - Computer-Assisted Music Analysis Toolbox (CAMAT)"
__license__ = "Licensed under the MIT license:"
__version__ = "1.0"


from .parse import pianoroll_parts
from .analyse import ambitus,\
    pitch_histogram,\
    pitch_class_histogram,\
    quarterlength_duration_histogram,\
    interval,\
    metric_profile,\
    time_signature_histogram,\
    metric_profile_split_time_signature,\
    max_measure_num, count_notes
from .web_scrapper import get_file_from_server, \
    get_files_from_server

from .corpus import analyse_pitch_class, analyse_interval, analyse_basic_statistics
from .search import simple_interval_search