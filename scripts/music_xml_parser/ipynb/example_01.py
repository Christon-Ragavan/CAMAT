import os
import sys
import matplotlib
sys.path.append(os.getcwd().replace(os.path.join('music_xml_parser', 'ipynb'), ''))

import music_xml_parser

from music_xml_parser.core import parse

xml_file = 'BrumAn_Bru1011_COM_3-6_MissaProde_002_01134.xml'

d = parse.with_xml_file(file_name=xml_file, plot_pianoroll=True,
                  save_at=None, save_file_name=None,
                  do_save=False)