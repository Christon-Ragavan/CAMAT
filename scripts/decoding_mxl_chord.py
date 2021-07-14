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


def generate_row(mus_object, part):
    d = {}

    d.update({'id': mus_object.id,
              'Part Name': part.partName,
              'Chord': mus_object,
              'Duration': mus_object.duration.quarterLength})
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

def xml_to_list_stream(xml_data):


    xml_list = []
    for part in xml_data.parts:

        instrument = part.getInstrument().instrumentName
        print("--------------------")
        print(instrument)
        print("--------------------")
        part.str

def xml_to_list(xml_data):

    """
    onset, duration, pitch_s, volume, instrument, event

    :param xml_data:
    :return:
    """
    xml_list = []
    for part in xml_data.parts:

        instrument = part.getInstrument().instrumentName
        print("----------------------------")
        print(instrument)
        print("----------------------------")
        # elt = part.notesAndRests
        # elt = part.flat
        elt = part.flat.elements

        # print(elt[0])
        for e in elt:

            start = e.offset
            duration = e.quarterLength
            print("-- el: ", e, type(e))
            print("-- -el: ", start, duration)


            try:
                if e.isChord:
                    event = 'Chord'
                    print("Chord")
                    for chord_note in e.pitches:
                        pitch = chord_note.ps
                        volume = e.volume.realized
                        xml_list.append([start, duration, pitch, volume, instrument, event])

                elif e.isNote:
                    event = 'Note'
                    print("Note")
                    pitch = e.pitch.ps
                    volume = e.volume.realized
                    xml_list.append([start, duration, pitch, volume, instrument, event])
                elif e.isRest:
                    event = 'Rest'
                    pitch, volume = 'NA', 'NA'

                    xml_list.append([start, duration, pitch, volume, instrument, event])

                    print("Rest")

                else:

                    pass

                    # pitch = note.pitch.ps
                    # volume = note.volume.realized
                    # xml_list.append([start, duration, pitch, volume, instrument])
            except:
                print("-- skipped")
                continue
            # xml_list.append([start, duration, pitch, rest, volume, instrument])
        #
        #     xml_list.append([onset, duration, pitch_s, volume, instrument, event])
        #
        #
        # for note in part.flat.notes:
        #     if note.isChord:
        #         print("True")
        #         start = note.offset
        #         duration = note.quarterLength
        #
        #         for chord_note in note.pitches:
        #             pitch = chord_note.ps # this is a shortcut for pitch.Pitch
        #             volume = note.volume.realized
        #             xml_list.append([start, duration, pitch, volume, instrument])
        #
        #     else:
        #         start = note.offset
        #         duration = note.quarterLength
        #         pitch = note.pitch.ps
        #         volume = note.volume.realized
        #         xml_list.append([start, duration, pitch, volume, instrument])

    # xml_list = sorted(xml_list, key=lambda x: (x[0], x[2]))
    df_xml = pd.DataFrame(xml_list, columns=['Start', 'Duration', 'Pitch', 'volume', 'Instrument', 'Rest'])
    return df_xml
"""
   Start  Duration  Pitch   volume Instrument Event     Measure_p1 Measure_p2    
0  0.000     1.000   68.0  0.70866      Piano Chord     1           1
1  0.000     1.000   76.0  0.70866      Piano Note      1           1
2  0.000     1.000   NA    NA           Piano Rest      1           2
2  0.000     1.000   NA    NA           Piano Rest      2           2
2  0.000     1.000   NA    NA           Piano Rest      2           2


N harmonic (exchange) : note can be represented in differnet ways 
was this note, chord or rest,
"""

def generate_df(score):
    parts = score.parts
    rows_list = []

    for part in parts:
        # for index, elt in enumerate(part.flat.stripTies(retainContainers=True).getElementsByClass([chord.Chord])):
        # for index, elt in pa:
            # print("--       :",elt)

            # print("-- offset: ", elt.offset)
            # print("-- duration: ", elt.duration.quarterLength)
            # if hasattr(elt, 'pitches'):
            #     pitches = elt.pitches
            #     for pitch in pitches:
            #         rows_list.append(generate_row(elt, part, pitch.pitchClass))
            # else:
            rows_list.append(generate_row(elt, part))
    return pd.DataFrame(rows_list)

def pitch_to_midi(np_o_pc):
    midi = [_pitch_to_midi(np_o_pc[i,0], np_o_pc[i,1]) for i in range(np.shape(np_o_pc)[0])]
    return np.asarray(midi, dtype=int)

def _get_file():
    if True:
        # print('Path to music21 environment', us_path)
        b = corpus.parse('bach/bwv66.6')
    else:
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_multi_accedentials.mxl"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_poly.mxl"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/testcase_poly_xml.xml"
        path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_ives1.xml"

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
    # df_d = generate_df(b)
    df_d = xml_to_list(b)
    # df_d = xml_to_list_stream(b)
    print(df_d)
    # print(df_d.to_string())
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



if __name__=="__main__":
    main()




