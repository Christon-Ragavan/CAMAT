#!/usr/bin/python

# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license
# Copyright (c) 2021 www.hfm-weimar.de  Hochschule für Musik FRANZ LISZT Weimar

__author__ = "Christon Nadar"
__copyright__ = "Copyright (c) 2021 www.hfm-weimar.de  Hochschule für Musik FRANZ LISZT Weimar"
__license__ = "Licensed under the MIT license:"
__version__ = "1.0"


from .parse import with_xml_file, \
    pianoroll_parts
from .analyse import ambitus,\
    pitch_histogram,\
    pitch_class_histogram,\
    quarterlength_duration_histogram,\
    interval,\
    metric_profile,\
    time_signature_histogram,\
    metric_profile_split_time_signature, max_measure_num
from .utils import export_as_csv,display_table

from .web_scrapper import get_file_from_server, \
    get_files_from_server
