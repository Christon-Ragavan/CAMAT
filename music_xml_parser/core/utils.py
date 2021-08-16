import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import HTML, display
import os

def export_as_csv(data, columns, save_file_name :str=None, do_save=False, do_print=False, do_return_pd=False, sep=',', index=False, header=True):
    """
    data (list): nd array as list
    columns (list): list of column header in strings
    save_at (str) : path the csv to be saved
    """
    if '.csv' not in  save_file_name:
        save_file_name = save_file_name+'.csv'
    if save_file_name is None:
        do_save=False
    base_die = os.getcwd().replace('core', os.path.join('data', 'exports'))

    save_at = os.path.join(base_die, save_file_name)
    pd_data = pd.DataFrame(data, columns=columns)
    pd_data.to_csv(save_at, sep=';', index=index, header=header)
    if do_print:
        display(HTML(pd_data.to_html(index=False)))
    if do_return_pd:
        return pd_data

def display_table(data, columns, do_return_pd=False, sep=',', index=False, header=True):

    pd_data = pd.DataFrame(data, columns=columns)
    display(HTML(pd_data.to_html(index=False)))
    if do_return_pd:
        return pd_data

