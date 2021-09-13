"""
Author: Christon Nadar
License: The MIT license, https://opensource.org/licenses/MIT
"""

import logging
from os.path import join, basename, isfile
import os

def _get_file_path(file_name):

    file = join(os.getcwd().replace(basename(os.getcwd()), 'data'), join('xmls_to_parse', 'xml_pool', file_name))
    if not isfile(file):
        file = join(os.getcwd().replace(basename(os.getcwd()), 'data'), join('xmls_to_parse', 'hfm_database', file_name))
    assert isfile(file), f"FIle Not Found {file}"
    return file

def set_up_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s')
    file_handler = logging.FileHandler('logs.log', 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stream_formatter = logging.Formatter('%(levelname)s:%(name)s:%(lineno)d:%(message)s')
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(stream_formatter)
    # logger.addHandler(stream_handler)
    return logger


def file_path_handler(file_path):

    pass

def save_df_xml(save_at):

    pass
class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None

    def zoom_factory(self, ax, base_scale=2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            try:
                relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            except:
                pass
            # rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
            # ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])

            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event', onPress)
        fig.canvas.mpl_connect('button_release_event', onRelease)
        fig.canvas.mpl_connect('motion_notify_event', onMotion)

        # return the function
        return onMotion
