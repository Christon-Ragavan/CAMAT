import itertools as it
import os

import numpy as np
import pandas as pd
from IPython.display import display, HTML
import warnings

np.seterr(all="ignore")

def _gen_alpha(N):
    letters = []
    for i in range(N):
        newCharacter = i % 26
        i //= 26
        s = "" + chr(newCharacter + ord('A'))
        while i != 0:
            newCharacter = i % 26
            i //= 26
            s = chr(newCharacter + ord('A')) + s
        letters.append(s)
    return letters

def export_as_csv(data, columns,
                  save_file_name :str=None,
                  do_print=False,
                  do_return_pd=False,
                  do_save =True,
                  sep=';',
                  index=False,
                  header=True,
                  save_at=None):
    """
    data (list): nd array as list
    columns (list): list of column header in strings
    save_at (str) : path the csv to be saved
    """
    N = np.shape(data)[1]
    if len(columns) != N:
        warnings.warn(
            f"Number of data column is not equal to the column names.\nExpected {N} column names but got {len(columns)}: {columns} instead. \nSetting default column names")
        columns = _gen_alpha(N=N)
    if '.csv' not in  save_file_name:
        save_file_name = save_file_name+'.csv'
    if save_at==None:
        base_die = os.getcwd().replace('ipynb', os.path.join('data', 'exports'))
        save_at = os.path.join(base_die, save_file_name)
    pd_data = pd.DataFrame(data, columns=columns)
    pd_data.to_csv(save_at, sep=sep, index=index, header=header)
    if do_print:
        display(HTML(pd_data.to_html(index=False)))
    if do_return_pd:
        return pd_data

def display_table(data, columns, do_return_pd=False, sep=',', index=False, header=True):
    try:
        N = np.shape(data)[1]
        if len(columns) != N:
            warnings.warn(
                f"Number of data column is not equal to the column names.\nExpected {N} column names but got {len(columns)}: {columns} instead. \nSetting default column names")
            columns = _gen_alpha(N=N)

        if isinstance(data, pd.DataFrame):
            pd_data = data
            pd_data.columns = columns
        else:
            pd_data = pd.DataFrame(data, columns=columns)

        display(HTML(pd_data.to_html(index=False)))
        if do_return_pd:
            return pd_data
    except:
        print("Check the input data")
        pass


def str2midi(note_string):
  """
  This fuction (str2midi) is taken from: lazy_midi
  https://github.com/danilobellini/audiolazy/blob/master/audiolazy/lazy_midi.py
  Given a note string name (e.g. "Bb4"), returns its MIDI pitch number.
  """
  MIDI_A4 = 69

  if note_string == "?":
    return np.nan
  data = note_string.strip().lower()
  name2delta = {"c": -9, "d": -7, "e": -5, "f": -4, "g": -2, "a": 0, "b": 2}
  accident2delta = {"b": -1, "#": 1, "x": 2}
  accidents = list(it.takewhile(lambda el: el in accident2delta, data[1:]))
  octave_delta = int(data[len(accidents) + 1:]) - 4
  return (MIDI_A4 +
          name2delta[data[0]] + # Name
          sum(accident2delta[ac] for ac in accidents) + # Accident
          12 * octave_delta # Octave
         )


def midi2str(midi_number, sharp=True):
  """
  This fuction (midi2str) is taken from: lazy_midi
  https://github.com/danilobellini/audiolazy/blob/master/audiolazy/lazy_midi.py
  Given a MIDI pitch number, returns its note string name (e.g. "C3").
  """
  MIDI_A4 = 69
  if np.isinf(midi_number) or np.isnan(midi_number):
    return "?"
  num = midi_number - (MIDI_A4 - 4 * 12 - 9)
  note = (num + .5) % 12 - .5
  rnote = int(round(note))
  error = note - rnote
  octave = str(int(round((num - note) / 12.)))
  if sharp:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
  else:
    names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
  names = names[rnote] + octave
  if abs(error) < 1e-4:
    return names
  else:
    err_sig = "+" if error > 0 else "-"
    err_str = err_sig + str(round(100 * abs(error), 2)) + "%"
    return names + err_str


def midi2pitchclass(midi_number, sharp=True):
    MIDI_A4 = 69
    if np.isinf(midi_number) or np.isnan(midi_number):
        return "?"
    num = midi_number - (MIDI_A4 - 4 * 12 - 9)
    note = (num + .5) % 12 - .5
    rnote = int(round(note))
    if sharp:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    else:
        names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return names[rnote], rnote

def pitchclassid2pitchclass(id, sharp=True):
    assert type(id) ==int
    if sharp:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    else:
        names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return names[id]

def _shift_local_onset(df):
    local_o = df['LocalOnset'].tolist()
    local_o = [round(float(i)+1,2) for i in local_o]
    df.drop('LocalOnset', axis=1, inplace=True)
    df['LocalOnset'] = local_o
    return df

def print_full_df(x):
    if len(x)==0:
        print("Empty list")
    else:
        pd.set_option('display.max_rows', len(x))
        display(HTML(x.to_html()))
        # pd.reset_option('display.max_rows')
        pd.set_option('display.max_rows', 10)

