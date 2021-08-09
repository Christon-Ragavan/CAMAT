import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import HTML, display


def export_as_csv(data, columns, save_at, do_print=False, do_return_pd=False, sep=',', index=False, header=True):
    """
    data (list): nd array as list
    columns (list): list of column header in strings
    save_at (str) : path the csv to be saved
    """

    pd_data = pd.DataFrame(data, columns=columns)
    pd_data.to_csv(save_at, sep=';', index=index, header=header)
    if do_print:
        display(HTML(pd_data.to_html(index=False)))
    if do_return_pd:
        return pd_data

