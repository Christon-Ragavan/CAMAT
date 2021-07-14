from music21 import *
from music21 import pitch
import os
import music21 as m21
import xml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# print
us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()

us['musescoreDirectPNGPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['musicxmlPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'


def generate_row(mus_object, part, pitch_class=np.nan):
    d = {}

    d.update({'Offset': mus_object.offset,
              'Duration': mus_object.duration.quarterLength,
              'Pitch': mus_object.pitch,
              'Octave': mus_object.pitch.octave,
              'Pitch Class': pitch_class})

    """
    d.update({'id': mus_object.id,
              'Offset': mus_object.offset,
              'Pitch': mus_object.pitch,
              'Octave': mus_object.pitch.octave,
              'Chord': mus_object.chord.Chord,
              'Duration': mus_object.duration.quarterLength,
              'Type': type(mus_object),
              'Pitch Class': pitch_class})
    
    """
    return d


def _pitch_to_midi(octave, pitch_class):
    o = (octave+1)*12
    m = pitch_class+o
    return m

def _create_dict():
    dict_12_pc = {'C':0,
                  'C#':1,
                  'D':2, 'D#':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}


    return dict_12_pc

def generate_df(score):
    parts = score.parts
    rows_list = []
    for part in parts:
        for index, elt in enumerate(part.flat.stripTies(retainContainers=True).getElementsByClass([note.Note])):
            # print(elt)
            if hasattr(elt, 'pitches'):
                pitches = elt.pitches
                for pitch in pitches:
                    rows_list.append(generate_row(elt, part, pitch.pitchClass))
            else:
                rows_list.append(generate_row(elt, part))
        for index, elt in enumerate(part.flat.stripTies(retainContainers=True)):
            print(elt, elt.duration)

    return pd.DataFrame(rows_list)

def pitch_to_midi(np_o_pc):
    midi = [_pitch_to_midi(np_o_pc[i,0], np_o_pc[i,1]) for i in range(np.shape(np_o_pc)[0])]
    return np.asarray(midi, dtype=int)

def _get_file():
    if False:
        # print('Path to music21 environment', us_path)
        b = corpus.parse('bach/bwv66.6')
    else:
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_multi_accedentials.mxl"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_poly.mxl"
        path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_ives1.xml"


        # MAC Os
        path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/ultimate_tie_test3.xml"
        # path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/test_case_poly.mxl"
        # path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/test_case_ives1.xml"

        b = converter.parse(path)
        b.show('text')
    return b

def plot_hist(df_p):
    unique, values = np.unique(df_p, return_counts=True)

    fig, ax = plt.subplots()
    note = [str(i) for i in unique]
    # x_pos = np.


    ax.bar(x_pos, values, align='center',
           color='salmon', ecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(note)
    ax.set_title('Histogram')
    plt.show()

def main():
    b = _get_file()
    df_d = generate_df(b)
    #
    print(df_d.to_string())
    # print(df_d[["Octave", "Pitch Class"]])
    #
    # df_o_pc = df_d[["Octave", "Pitch Class"]]
    #
    # np_o_pc = df_o_pc.to_numpy()
    # midi = pitch_to_midi(np_o_pc)
    # print(len(midi), np.shape(np_o_pc))
    # assert np.shape(midi)[0] == np.shape(np_o_pc)[0]
    # df_d['Midi'] = pd.Series(midi, index=df_d.index)
    # print(df_d.to_string())
    # print(df_d[["Midi"]])
    # df_m = df_d[["Midi"]]
    # df_np_m = df_m.to_numpy()
    # # plot_hist(df_np_m)

"""
1. try multiple loops - note and chord seperatly 
2. try getElementbyTag method how it works - search if anyother xml parsing methods are available 
3. try to get other info - 'Start', 'Duration', 'MIDI Pitch', 'Pitch', 'Voice', 'Instrument', 'Event','Volume'

"""

if __name__=="__main__":
    main()
