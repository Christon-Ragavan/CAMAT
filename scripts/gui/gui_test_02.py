# Author: Miguel Martinez Lopez
# Version: 0.20
import re
import sys

import numpy as np

sys.path.append('/Users/chris/DocumentLocal/workspace')
try:
    from Tkinter import Frame, Label, Message, StringVar, Canvas
    from ttk import Scrollbar
    from Tkconstants import *
    import Tkinter as tk

except ImportError:
    from tkinter import Frame, Label, Message, StringVar, Canvas
    from tkinter.ttk import Scrollbar
    from tkinter.constants import *
    import tkinter as tk

import platform
from functools import partial
from hfm.scripts_in_progress.xml_parser.scripts.hfm_database_search import run_search

OS = platform.system()


class Mousewheel_Support(object):
    # implemetation of singleton pattern
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, root, horizontal_factor=2, vertical_factor=2):

        self._active_area = None

        if isinstance(horizontal_factor, int):
            self.horizontal_factor = horizontal_factor
        else:
            raise Exception("Vertical factor must be an integer.")

        if isinstance(vertical_factor, int):
            self.vertical_factor = vertical_factor
        else:
            raise Exception("Horizontal factor must be an integer.")

        if OS == "Linux":
            root.bind_all('<4>', self._on_mousewheel, add='+')
            root.bind_all('<5>', self._on_mousewheel, add='+')
        else:
            # Windows and MacOS
            root.bind_all("<MouseWheel>", self._on_mousewheel, add='+')

    def _on_mousewheel(self, event):
        if self._active_area:
            self._active_area.onMouseWheel(event)

    def _mousewheel_bind(self, widget):
        self._active_area = widget

    def _mousewheel_unbind(self):
        self._active_area = None

    def add_support_to(self, widget=None, xscrollbar=None, yscrollbar=None, what="units", horizontal_factor=None,
                       vertical_factor=None):
        if xscrollbar is None and yscrollbar is None:
            return

        if xscrollbar is not None:
            horizontal_factor = horizontal_factor or self.horizontal_factor

            xscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget, 'x', self.horizontal_factor, what)
            xscrollbar.bind('<Enter>', lambda event, scrollbar=xscrollbar: self._mousewheel_bind(scrollbar))
            xscrollbar.bind('<Leave>', lambda event: self._mousewheel_unbind())

        if yscrollbar is not None:
            vertical_factor = vertical_factor or self.vertical_factor

            yscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget, 'y', self.vertical_factor, what)
            yscrollbar.bind('<Enter>', lambda event, scrollbar=yscrollbar: self._mousewheel_bind(scrollbar))
            yscrollbar.bind('<Leave>', lambda event: self._mousewheel_unbind())

        main_scrollbar = yscrollbar if yscrollbar is not None else xscrollbar

        if widget is not None:
            if isinstance(widget, list) or isinstance(widget, tuple):
                list_of_widgets = widget
                for widget in list_of_widgets:
                    widget.bind('<Enter>', lambda event: self._mousewheel_bind(widget))
                    widget.bind('<Leave>', lambda event: self._mousewheel_unbind())

                    widget.onMouseWheel = main_scrollbar.onMouseWheel
            else:
                widget.bind('<Enter>', lambda event: self._mousewheel_bind(widget))
                widget.bind('<Leave>', lambda event: self._mousewheel_unbind())

                widget.onMouseWheel = main_scrollbar.onMouseWheel

    @staticmethod
    def _make_mouse_wheel_handler(widget, orient, factor=1, what="units"):
        view_command = getattr(widget, orient + 'view')

        if OS == 'Linux':
            def onMouseWheel(event):
                if event.num == 4:
                    view_command("scroll", (-1) * factor, what)
                elif event.num == 5:
                    view_command("scroll", factor, what)

        elif OS == 'Windows':
            def onMouseWheel(event):
                view_command("scroll", (-1) * int((event.delta / 120) * factor), what)

        elif OS == 'Darwin':
            def onMouseWheel(event):
                view_command("scroll", event.delta, what)

        return onMouseWheel


class Scrolling_Area(Frame, object):

    def __init__(self, master, width=None, anchor=N, height=None, mousewheel_speed=2, scroll_horizontally=True,
                 xscrollbar=None, scroll_vertically=True, yscrollbar=None, outer_background=None, inner_frame=Frame,
                 **kw):
        Frame.__init__(self, master, class_=self.__class__)

        if outer_background:
            self.configure(background=outer_background)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._width = width
        self._height = height

        self.canvas = Canvas(self, background=outer_background, highlightthickness=0, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky=N + E + W + S)

        if scroll_vertically:
            if yscrollbar is not None:
                self.yscrollbar = yscrollbar
            else:
                self.yscrollbar = Scrollbar(self, orient=VERTICAL)
                self.yscrollbar.grid(row=0, column=1, sticky=N + S)

            self.canvas.configure(yscrollcommand=self.yscrollbar.set)
            self.yscrollbar['command'] = self.canvas.yview
        else:
            self.yscrollbar = None

        if scroll_horizontally:
            if xscrollbar is not None:
                self.xscrollbar = xscrollbar
            else:
                self.xscrollbar = Scrollbar(self, orient=HORIZONTAL)
                self.xscrollbar.grid(row=1, column=0, sticky=E + W)

            self.canvas.configure(xscrollcommand=self.xscrollbar.set)
            self.xscrollbar['command'] = self.canvas.xview
        else:
            self.xscrollbar = None

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.innerframe = inner_frame(self.canvas, **kw)
        self.innerframe.pack(anchor=anchor)

        self.canvas.create_window(0, 0, window=self.innerframe, anchor='nw', tags="inner_frame")

        self.canvas.bind('<Configure>', self._on_canvas_configure)

        Mousewheel_Support(self).add_support_to(self.canvas, xscrollbar=self.xscrollbar, yscrollbar=self.yscrollbar)

    @property
    def width(self):
        return self.canvas.winfo_width()

    @width.setter
    def width(self, width):
        self.canvas.configure(width=width)

    @property
    def height(self):
        return self.canvas.winfo_height()

    @height.setter
    def height(self, height):
        self.canvas.configure(height=height)

    def set_size(self, width, height):
        self.canvas.configure(width=width, height=height)

    def _on_canvas_configure(self, event):
        width = max(self.innerframe.winfo_reqwidth(), event.width)
        height = max(self.innerframe.winfo_reqheight(), event.height)

        self.canvas.configure(scrollregion="0 0 %s %s" % (width, height))
        self.canvas.itemconfigure("inner_frame", width=width, height=height)

    def update_viewport(self):
        self.update()

        window_width = self.innerframe.winfo_reqwidth()
        window_height = self.innerframe.winfo_reqheight()

        if self._width is None:
            canvas_width = window_width
        else:
            canvas_width = min(self._width, window_width)

        if self._height is None:
            canvas_height = window_height
        else:
            canvas_height = min(self._height, window_height)

        self.canvas.configure(scrollregion="0 0 %s %s" % (window_width, window_height), width=canvas_width,
                              height=canvas_height)
        self.canvas.itemconfigure("inner_frame", width=window_width, height=window_height)


class Cell(Frame):
    """Base class for cells"""


class Data_Cell(Cell):
    def __init__(self, master, variable, anchor=W, bordercolor=None, borderwidth=1, padx=0, pady=0, background=None,
                 foreground=None, font=None):
        Cell.__init__(self, master, background=background, highlightbackground=bordercolor, highlightcolor=bordercolor,
                      highlightthickness=borderwidth, bd=0)

        self._message_widget = Message(self, textvariable=variable, font=font, background=background,
                                       foreground=foreground)
        self._message_widget.pack(expand=True, padx=padx, pady=pady, anchor=anchor)


class Header_Cell(Cell):
    def __init__(self, master, text, bordercolor=None, borderwidth=1, padx=0, pady=0, background=None, foreground=None,
                 font=None, anchor=CENTER, separator=True):
        Cell.__init__(self, master, background=background, highlightbackground=bordercolor, highlightcolor=bordercolor,
                      highlightthickness=borderwidth, bd=0)
        self.pack_propagate(False)

        self._header_label = Label(self, text=text, background=background, foreground=foreground, font=font)
        self._header_label.pack(padx=padx, pady=pady, expand=True)

        if separator and bordercolor is not None:
            separator = Frame(self, height=2, background=bordercolor, bd=0, highlightthickness=0, class_="Separator")
            separator.pack(fill=X, anchor=anchor)

        self.update()
        height = self._header_label.winfo_reqheight() + 2 * padx
        width = self._header_label.winfo_reqwidth() + 2 * pady

        self.configure(height=height, width=width)


class Table(Frame):
    def __init__(self, master, columns, column_weights=None, column_minwidths=None, height=500, minwidth=20,
                 minheight=20, padx=5, pady=5, cell_font=None, cell_foreground="black", cell_background="white",
                 cell_anchor=W, header_font=None, header_background="white", header_foreground="black",
                 header_anchor=CENTER, bordercolor="#999999", innerborder=True, outerborder=True,
                 stripped_rows=("#EEEEEE", "white"), on_change_data=None, mousewheel_speed=2, scroll_horizontally=False,
                 scroll_vertically=True):
        outerborder_width = 1 if outerborder else 0

        Frame.__init__(self, master, bd=0)

        self._cell_background = cell_background
        self._cell_foreground = cell_foreground
        self._cell_font = cell_font
        self._cell_anchor = cell_anchor

        self._stripped_rows = stripped_rows

        self._padx = padx
        self._pady = pady

        self._bordercolor = bordercolor
        self._innerborder_width = 1 if innerborder else 0

        self._data_vars = []

        self._columns = columns

        self._number_of_rows = 0
        self._number_of_columns = len(columns)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._head = Frame(self, highlightbackground=bordercolor, highlightcolor=bordercolor,
                           highlightthickness=outerborder_width, bd=0)
        self._head.grid(row=0, column=0, sticky=E + W)

        header_separator = False if outerborder else True

        for j in range(len(columns)):
            column_name = columns[j]

            header_cell = Header_Cell(self._head, text=column_name, borderwidth=self._innerborder_width,
                                      font=header_font, background=header_background, foreground=header_foreground,
                                      padx=padx, pady=pady, bordercolor=bordercolor, anchor=header_anchor,
                                      separator=header_separator)
            header_cell.grid(row=0, column=j, sticky=N + E + W + S)

        add_scrollbars = scroll_horizontally or scroll_vertically
        if add_scrollbars:
            if scroll_horizontally:
                xscrollbar = Scrollbar(self, orient=HORIZONTAL)
                xscrollbar.grid(row=2, column=0, sticky=E + W)
            else:
                xscrollbar = None

            if scroll_vertically:
                yscrollbar = Scrollbar(self, orient=VERTICAL)
                yscrollbar.grid(row=1, column=1, sticky=N + S)
            else:
                yscrollbar = None

            scrolling_area = Scrolling_Area(self, width=self._head.winfo_reqwidth(), height=height,
                                            scroll_horizontally=scroll_horizontally, xscrollbar=xscrollbar,
                                            scroll_vertically=scroll_vertically, yscrollbar=yscrollbar)
            scrolling_area.grid(row=1, column=0, sticky=E + W)

            self._body = Frame(scrolling_area.innerframe, highlightbackground=bordercolor, highlightcolor=bordercolor,
                               highlightthickness=outerborder_width, bd=0)
            self._body.pack()

            def on_change_data():
                scrolling_area.update_viewport()

        else:
            self._body = Frame(self, height=height, highlightbackground=bordercolor, highlightcolor=bordercolor,
                               highlightthickness=outerborder_width, bd=0)
            self._body.grid(row=1, column=0, sticky=N + E + W + S)

        if column_weights is None:
            for j in range(len(columns)):
                self._body.grid_columnconfigure(j, weight=1)
        else:
            for j, weight in enumerate(column_weights):
                self._body.grid_columnconfigure(j, weight=weight)

        if column_minwidths is not None:
            for j, minwidth in enumerate(column_minwidths):
                if minwidth is None:
                    header_cell = self._head.grid_slaves(row=0, column=j)[0]
                    minwidth = header_cell.winfo_reqwidth()

                self._body.grid_columnconfigure(j, minsize=minwidth)
        else:
            for j in range(len(columns)):
                header_cell = self._head.grid_slaves(row=0, column=j)[0]
                minwidth = header_cell.winfo_reqwidth()

                self._body.grid_columnconfigure(j, minsize=minwidth)

        self._on_change_data = on_change_data

    def _append_n_rows(self, n):
        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        for i in range(number_of_rows, number_of_rows + n):
            list_of_vars = []
            for j in range(number_of_columns):
                var = StringVar()
                list_of_vars.append(var)

                if self._stripped_rows:
                    cell = Data_Cell(self._body, borderwidth=self._innerborder_width, variable=var,
                                     bordercolor=self._bordercolor, padx=self._padx, pady=self._pady,
                                     background=self._stripped_rows[i % 2], foreground=self._cell_foreground,
                                     font=self._cell_font, anchor=self._cell_anchor)
                else:
                    cell = Data_Cell(self._body, borderwidth=self._innerborder_width, variable=var,
                                     bordercolor=self._bordercolor, padx=self._padx, pady=self._pady,
                                     background=self._cell_background, foreground=self._cell_foreground,
                                     font=self._cell_font, anchor=self._cell_anchor)

                cell.grid(row=i, column=j, sticky=N + E + W + S)

            self._data_vars.append(list_of_vars)

        if number_of_rows == 0:
            for j in range(self.number_of_columns):
                header_cell = self._head.grid_slaves(row=0, column=j)[0]
                data_cell = self._body.grid_slaves(row=0, column=j)[0]
                data_cell.bind("<Configure>",
                               lambda event, header_cell=header_cell: header_cell.configure(width=event.width), add="+")

        self._number_of_rows += n

    def _pop_n_rows(self, n):
        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        for i in range(number_of_rows - n, number_of_rows):
            for j in range(number_of_columns):
                self._body.grid_slaves(row=i, column=j)[0].destroy()

            self._data_vars.pop()

        self._number_of_rows -= n

    def set_data(self, data):
        n = len(data)
        m = len(data[0])

        number_of_rows = self._number_of_rows

        if number_of_rows > n:
            self._pop_n_rows(number_of_rows - n)
        elif number_of_rows < n:
            self._append_n_rows(n - number_of_rows)

        for i in range(n):
            for j in range(m):
                self._data_vars[i][j].set(data[i][j])

        if self._on_change_data is not None: self._on_change_data()

    def get_data(self):
        number_of_rows = self._number_of_rows
        number_of_columns = self.number_of_columns

        data = []
        for i in range(number_of_rows):
            row = []
            row_of_vars = self._data_vars[i]
            for j in range(number_of_columns):
                cell_data = row_of_vars[j].get()
                row.append(cell_data)

            data.append(row)
        return data

    @property
    def number_of_rows(self):
        return self._number_of_rows

    @property
    def number_of_columns(self):
        return self._number_of_columns

    def row(self, index, data=None):
        if data is None:
            row = []
            row_of_vars = self._data_vars[index]

            for j in range(self.number_of_columns):
                row.append(row_of_vars[j].get())

            return row
        else:
            number_of_columns = self.number_of_columns

            if len(data) != number_of_columns:
                raise ValueError("data has no %d elements: %s" % (number_of_columns, data))

            row_of_vars = self._data_vars[index]
            for j in range(number_of_columns):
                row_of_vars[index][j].set(data[j])

            if self._on_change_data is not None: self._on_change_data()

    def column(self, index, data=None):
        number_of_rows = self._number_of_rows

        if data is None:
            column = []

            for i in range(number_of_rows):
                column.append(self._data_vars[i][index].get())

            return column
        else:
            if len(data) != number_of_rows:
                raise ValueError("data has no %d elements: %s" % (number_of_rows, data))

            for i in range(number_of_columns):
                self._data_vars[i][index].set(data[i])

            if self._on_change_data is not None: self._on_change_data()

    def clear(self):
        number_of_rows = self._number_of_rows
        number_of_columns = self._number_of_columns

        for i in range(number_of_rows):
            for j in range(number_of_columns):
                self._data_vars[i][j].set("")

        if self._on_change_data is not None: self._on_change_data()

    def delete_row(self, index):
        i = index
        while i < self._number_of_rows:
            row_of_vars_1 = self._data_vars[i]
            row_of_vars_2 = self._data_vars[i + 1]

            j = 0
            while j < self.number_of_columns:
                row_of_vars_1[j].set(row_of_vars_2[j])

            i += 1

        self._pop_n_rows(1)

        if self._on_change_data is not None: self._on_change_data()

    def insert_row(self, data, index=END):
        self._append_n_rows(1)

        if index == END:
            index = self._number_of_rows - 1

        i = self._number_of_rows - 1
        while i > index:
            row_of_vars_1 = self._data_vars[i - 1]
            row_of_vars_2 = self._data_vars[i]

            j = 0
            while j < self.number_of_columns:
                row_of_vars_2[j].set(row_of_vars_1[j])
                j += 1
            i -= 1

        list_of_cell_vars = self._data_vars[index]
        for cell_var, cell_data in zip(list_of_cell_vars, data):
            cell_var.set(cell_data)

        if self._on_change_data is not None: self._on_change_data()

    def cell(self, row, column, data=None):
        """Get the value of a table cell"""
        if data is None:
            return self._data_vars[row][column].get()
        else:
            self._data_vars[row][column].set(data)
            if self._on_change_data is not None: self._on_change_data()

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, column = index
            return self.cell(row, column)
        else:
            raise Exception("Row and column indices are required")

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            row, column = index
            self.cell(row, column, value)
        else:
            raise Exception("Row and column indices are required")

    def on_change_data(self, callback):
        self._on_change_data = callback

def _Get_search(c, t , k ,mn, lty, ltr, y):
    """
        Composer': list, (example: ['Composer1', 'Composer2'])
        'Movement Number': list, (example: ['Mov1', 'Mov2'])
        'Title': list, (example: ['Title1', 'Title2'])
        'Key': list,  (example: ['C', 'G'])
        'Life Time Year': list, (example: [1300, 1600, 1880])
        'Life Time Range': str, (example: '1300-1600')
        'Year Range': str, (example: '1300-1600')
        """
    database_csv_path = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/database/'




    c = None if 'none' in c or 'None' in c else re.split(',', c)
    mn = None if 'none' in mn or 'None' in mn else re.split(',', mn)
    t = None if 'none' in t or 'None' in t else re.split(',', t)
    lty = None if 'none' in lty or 'None' in lty else re.split(',', lty)
    k = None if 'none' in k or 'None' in k else re.split(',', k)
    if 'none' in ltr or 'None' in ltr: ltr =None
    if 'none' in y or 'None' in y: y =None


    search_keywords = {'Composer': c,
                       'Movement Number': mn,
                       'Title': t,
                       'Key': k,
                       'Life Time Year': lty,
                       'Life Time Range': ltr,
                       'Year Range': y}
    print(search_keywords)
    df_s = run_search(search_keywords=search_keywords,
                      extract_database=False,
                      apply_precise_keyword_search=True,
                      extract_extire_database=False,
                      save_extracted_database_path=database_csv_path,
                      do_save_csv=False,
                      do_print=False,
                      save_search_output_path='_.csv')
    df_s = df_s.reset_index()
    print(df_s)
    list_df = df_s.values.tolist()


    columns= list(df_s.columns.values)
    c_l = []
    for l in columns:
        if "Movement" in l:
            a = "Movement Nr."
        else:
            a = l
        c_l.append(a)

    print(c_l)
    return c_l, list_df

class UI_interact():
    def __init__(self, root,canvas1):
        self.root = root
        self.canvas1 = canvas1

    def clear_button(self, data):
        self.table.forget()


    def search_button(self, data):
        try:
            self.clear_button(data)
        except:
            pass
        c = composer_e.get()
        mn = m_nr_e.get()
        t = title_e.get()
        lty = life_time_year_e.get()
        ltr = life_time_range_e.get()
        y = year_range_e.get()
        k = k_e.get()


        cols, list_df = _Get_search(c, t , k ,mn, lty, ltr, y)
        if len(list_df) ==0:
            list_df = [['NA']*len(cols)]

        self.table = Table(self.root, cols, column_minwidths=[100]*len(cols))
        self.table.pack(padx=2, pady=2)
        self.table.set_data(list_df)

        self.root.update()
        self.root.geometry("%sx%s" % (self.root.winfo_reqwidth(), 10000))


if __name__ == "__main__":

    #c = _Get_search('bach')

    try:
        from Tkinter import Tk

    except ImportError:
        from tkinter import Tk

    root = Tk()
    root.title('HFM Database Corpus Query')

    root.geometry("%sx%s" % (900, 300))

    canvas1 = tk.Canvas(root, width=900, height=300, relief='raised')
    canvas1.pack()

    label1 = tk.Label(root, text='Search hfm Database')
    label1.config(font=('helvetica', 14))
    canvas1.create_window(350, 25, window=label1)

    composer_var = tk.StringVar(root, value=None)
    composer_var.set("Jos")


    title_var = tk.StringVar(root, value=None)
    title_var.set("missa")

    key_var = tk.StringVar(root, value=None)
    key_var.set('None')

    m_nr_var = tk.StringVar(root, value=None)
    m_nr_var.set("1")


    life_time_year_var = tk.StringVar(root, value=None)
    life_time_year_var.set('None')

    life_time_range_var = tk.StringVar(root, value=None)
    life_time_range_var.set('None')

    year_range_var = tk.StringVar(root, value=None)
    year_range_var.set('None')



    x = 200
    y = 45
    inc = 20
    composer_label = tk.Label(root, text='Composer', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=composer_label)
    composer_e = tk.Entry(root, textvariable=composer_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=composer_e)
    y += inc

    title_l = tk.Label(root, text='Title', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=title_l)
    title_e = tk.Entry(root, textvariable=title_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=title_e)
    y += inc

    k_l = tk.Label(root, text='Key', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=k_l)
    k_e = tk.Entry(root, textvariable=key_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=k_e)
    y += inc

    m_nr_l = tk.Label(root, text='Movement Nr.', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=m_nr_l)
    m_nr_e = tk.Entry(root, textvariable=m_nr_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=m_nr_e)
    y += inc

    life_time_year_l = tk.Label(root, text='Life Time Year', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=life_time_year_l)
    life_time_year_e = tk.Entry(root, textvariable=life_time_year_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=life_time_year_e)
    y += inc

    life_time_range_l = tk.Label(root, text='Life Time Range', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=life_time_range_l)
    life_time_range_e = tk.Entry(root, textvariable=life_time_range_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=life_time_range_e)
    y += inc

    year_range_l = tk.Label(root, text='Year Range', font=('calibre', 10, 'bold'))
    canvas1.create_window(x, y, window=year_range_l)
    year_range_e = tk.Entry(root, textvariable=year_range_var, font=('calibre', 10, 'normal'))
    canvas1.create_window(x+150, y, window=year_range_e)
    y += inc



    ui_int = UI_interact(root=root, canvas1=canvas1)

    data = 'Keyword: '
    action_with_arg = partial(ui_int.search_button, data)
    button1 = tk.Button(root, text='Search', command=action_with_arg, bg='brown', fg='gray',
                        font=('helvetica', 9, 'bold'))
    button1.place(x=x+50, y=y)

    action_with_arg = partial(ui_int.clear_button, data)
    button2 = tk.Button(root, text='Clear', command=action_with_arg, bg='brown', fg='gray',
                        font=('helvetica', 9, 'bold'))
    button2.place(x=x+200, y=y)

    canvas1.create_window(200, 180, window=button1)
    root.mainloop()
