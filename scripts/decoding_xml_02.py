
from music21 import *
import numpy as np
from music21 import pitch
import os
import music21 as m21
import xml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hfm.scripts_in_progress.examples.utils_func import _create_pianoroll
print(pd.__version__)
print(m21.__version__)
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)
# print
us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()

us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()
print('Path to music21 environment', us_path)
print(us)

us['musescoreDirectPNGPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['musicxmlPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'


"""
us['musescoreDirectPNGPath'] = r'/usr/bin/musescore'
us['musescoreDirectPNGPath'] = r'/usr/bin/musescore'
us['musicxmlPath'] = r'/usr/bin/musescore'

"""
"""   istie = False
                    l_offset, l_duration = [], []
                    if e.tie:
                        print("Note TIED", start, duration, e, e.tie.type)
                        if  e.tie.type == 'start':
                            l_offset.append(start)
                            l_duration.append(duration)
                            print("worked start")

                        if  e.tie.type == 'continue':
                            l_duration.append(duration)
                        if  e.tie.type == 'stop':
                            print("worked stop")

                        # save this note untill if find next time end
                        # istie=True
                    else:
"""

def generate_row(mus_object, part):
    d = {}
    d.update({'id': mus_object.id,
              'Part Name': part.partName,
              'Pitch': mus_object.pitch,
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

def _merge_measure(df_xml,measure_data):
    data_st = df_xml["Start"].to_numpy()
    num_c = int(np.shape(measure_data)[1]/2)
    add_measure = []
    measure_col_np = np.zeros((np.shape(data_st)[0], num_c))

    for n_c in range(num_c):
        s_mea = 'start_'+ str(n_c+1)
        m_mea = 'measure_'+ str(n_c+1)
        subset = measure_data[[s_mea, m_mea]]
        m_tuples = [list(x) for x in subset.to_numpy()]
        for m_t_id, m_t in enumerate (m_tuples):
            # print("m",m_t_id, m_tuples[m_t_id][0],m_tuples[m_t_id][1] )
            mn = m_tuples[m_t_id][0]
            if m_t_id + 1 == len(m_tuples):
                mx =data_st[-1]
                # mx = m_tuples[m_t_id][0] +10
            else:
                mx = m_tuples[m_t_id + 1][0]
            n_sel = np.where((data_st >= mn) & (data_st <mx))

            # n_sel = list(n_sel)
            # print(n_sel)
            try:
                for ns in n_sel:
                    measure_col_np[ns, n_c] = int(m_tuples[m_t_id][1])
            except:
                print("ERROR")
                continue

    pd_m = pd.DataFrame(measure_col_np, columns=["Measure_part"+str(i+1) for i in range(num_c)])
    merged_xml_data = pd.concat([df_xml, pd_m], axis=1)
    return merged_xml_data


def _get_measure_data(b):

    num_parts = len(b.getElementsByClass(stream.Part))
    measure_infos = []
    for i in range(num_parts):
        m_info = b.getElementsByClass(stream.Part)[i].getElementsByClass(stream.Measure)
        print(m_info)
        part_offset = []
        part_measurenum = []

        for measure_num, m in enumerate (m_info):
            print( f"Measure number {measure_num+1} offset {m.offset}")
            part_offset.append(m.offset)
            part_measurenum.append(measure_num+1)
        measure_infos.extend([part_offset, part_measurenum])
        # measure_infos.extend(part_offset)
        # measure_infos.extend(part_measurenum)
    m_infos_np = np.asarray(measure_infos).T
    col_labes_offset = ['start_'+str(i+1)for i in range(num_parts)]
    col_labes_measure = ['measure_'+str(i+1)for i in range(num_parts)]
    cols = []
    for i in range(num_parts):
        cols.append(col_labes_offset[i])
        cols.append(col_labes_measure[i])
    df_mea = pd.DataFrame(m_infos_np, columns=cols)
    #####################################################
    # Check for similarities TODO
    #####################################################
    return df_mea

def _convert_midi_pitch_class(df_xml_merged_measure):

    midi_pitch = df_xml_merged_measure["MIDI Pitch"].to_numpy()
    pitch_class = []

    for i in midi_pitch:
        if i == np.inf:
            pitch_class.append(np.inf)
        else:
            pitch_class.append(i%12)
    return pitch_class

def xml_to_list(xml_data):

    """
    onset, duration, pitch_s, volume, instrument, event

    :param xml_data:
    :return:
    """
    import re

    # measure_data = _get_measure_data()

    xml_list = []
    # meter = xml_data.meter.Ti

    measure_data = _get_measure_data(xml_data)

    for part in xml_data.parts:
        global_offset = []
        g_offset = 0
        firstnote = True
        instrument = part.getInstrument().instrumentName
        ins = str(part.getInstrument())
        print("----------------------------")
        print(instrument)
        print(part.getInstrument(), type(part.getInstrument()), re.split(':',ins ))
        print("----------------------------")
        v = re.split(':',ins )
        Voice = str(v[1]).replace(' ','')

        elt = part.flat.elements

        for idx, e in enumerate (elt):

            start = e.offset
            duration = e.quarterLength
            start = float(start)
            duration = float(duration)
            print(f"Measure offset {start} duration {duration} type{e}")

            try:

                if e.tie != None:
                    t = e.tie.type
                    # print("---TIE-",t)

                if e.isChord:
                    #if e.tie:
                       # print("Chord TIED ", start, duration, e, e.tie.type)
                    event = 'Chord'

                    for chord_note in e.pitches:
                        midi_pitch = chord_note.ps
                        pitch = chord_note
                        g_offset =start
                        print(f".. Loff: {start} Ldur {duration} G:{g_offset} :{pitch}")

                        volume = e.volume.realized

                        xml_list.append([start, duration, midi_pitch, pitch, Voice, instrument, event, volume])

                elif e.isNote:

                    event = 'Note'
                    midi_pitch = e.pitch.ps
                    pitch = e.pitch
                    volume = e.volume.realized
                    g_offset += start
                    print(f".. Loff: {start} Ldur {duration} G:{g_offset} :{pitch}")
                    xml_list.append([start, duration, midi_pitch, pitch, Voice, instrument, event, volume])


                elif e.isRest:

                    event = 'Rest'
                    midi_pitch, pitch, volume = np.inf,np.inf, 0
                    g_offset += start
                    print(f".. Loff: {start} Ldur {duration} G:{g_offset} :{pitch}")
                    xml_list.append([start, duration, midi_pitch, pitch, Voice, instrument, event, volume])
                elif e.isMeasure:
                    print("MEASURESADASDS")
                    event = 'isMeasure\n\n'
                    midi_pitch, pitch, volume = np.inf,np.inf, 0
                    xml_list.append([start, duration, midi_pitch, pitch, Voice, instrument, event, volume])
                else:
                    pass
            except:
                continue


    #xml_list = sorted(xml_list, key=lambda x: (x[0], x[2]))

    df_xml = pd.DataFrame(xml_list, columns=['Start', 'Duration', 'MIDI Pitch', 'Pitch', 'Voice', 'Instrument', 'Event','Volume'])
    df_xml_merged_measure = _merge_measure(df_xml,measure_data )
    # print("####",type(df_xml_merged_measure))

    pitch_class = _convert_midi_pitch_class(df_xml_merged_measure)
    df_pc = pd.DataFrame(pitch_class)
    df_xml_merged_measure.insert(loc=4,  column="Pitch Class", value=df_pc)
    return df_xml_merged_measure


def generate_df(score):
    parts = score.parts
    rows_list = []
    for part in parts:
            rows_list.append(generate_row(elt, part))
    return pd.DataFrame(rows_list)

def pitch_to_midi(np_o_pc):
    midi = [_pitch_to_midi(np_o_pc[i,0], np_o_pc[i,1]) for i in range(np.shape(np_o_pc)[0])]
    return np.asarray(midi, dtype=int)

def _get_file():
    if False:
        # print('Path to music21 environment', us_path)
        b = corpus.parse('bach/bwv66.6')
        b.show('text')

    else:
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_multi_accedentials.mxl"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/complex_test_case.xml"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/sacre_ts.mxl"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/sacred_xml.xml"

        # diffent measure and time signature
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_Castuski_1_time_sig_measure.xml"


        # rhythme structure
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_acc_triples_02.xml"

        # old testcase working tied notes
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_ives1.xml"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/ultimate_tie_test.xml"

        # Example both works
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_poly.mxl"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/testcase_poly_xml.xml"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/BaJoSe_BWV8_COM_6-6_ChoraleHer_TobisNo_00097.xml"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/Jos1102a-Missa_La_sol_fa_re_mi-Kyrie.musicxml"
        # path = "/home/chris/Documents/Workspace/weimar/example_files/test_case_tied_note.xml"
        #


        # MAC Os
        #path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/ultimate_tie_test3.xml"

        path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files/ultimate_tie_test3.xml"
        # path = "/Users/chris/DocumentLocal/workspace/hfm/weimar/example_files/test_case_poly.mxl"

        # b = converter.parse(path).stripTies()
        b = converter.parse(path)
        # b = corpus.parse('bwv66.6')

        b.show('text')
        # if True:
        #     numDs = 0
        #     for n in b.recurse().notes:
        #         print(n, "no")
        #
        #         if (n.tie and n.tie.type=='start'):
        #             print(n, "start")
        #
        #         if (n.tie and n.tie.type=='stop'):
        #             print(n, "stop")
        #             # numDs += 1
        #
        #         # if (n.tie is None or n.tie.type == 'start'):
        #     print(numDs)

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
def generate_measure(score):
    parts = score.parts
    rows_list = []
    for part in parts:
        for index, elt in enumerate(part.flat.stripTies(retainContainers=True).getElementsByClass(stream.Measure)):
            print("Measure -- ",elt)

            rows_list.append(generate_row(elt, part))
    return pd.DataFrame(rows_list)
def plot_pc_hist(df_d):
    np_pc = df_d["Pitch Class"].to_numpy()
    u_pc, v = np.unique(np_pc, return_counts=True)
    values = np.zeros(12)
    print(u_pc, values)
    for id, i in enumerate (u_pc):
        print(id, u_pc[id])
        if u_pc[id] == np.inf:
            continue
        print(values[int(u_pc[id])], int(u_pc[id]))
        values[int(u_pc[id])]= int(v[id])
    # print(values)
    assert len(values)==12, "len of Pitch Classes are not exactly 12"
    fig, ax = plt.subplots()
    note = ('c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b')

    x_pos = np.arange(len(note))

    ax.bar(x_pos, values, align='center',
           color='salmon', ecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(note)
    ax.set_title('Pitch Class Histogram')
    plt.show()



    pass
def main():
    # corpus.parse
    # bass = corpus.parse('bwv66.6').parts['bass']
    # bass.show('text')
    # clef.bestClef(bass)
    # agnus = corpus.parse('palestrina/Agnus_01')
    # agnus.show('text')
    # agnusSop = agnus.parts[0]
    # agnusSop.measures(1, 7)

    #
    b = _get_file()
    df_d = xml_to_list(b)
    print(df_d)
    # # plot_pc_hist(df_d)
    # plot_piano_roll(pitch=df_d["MIDI Pitch"].to_numpy(), time=df_d["Start"].to_numpy(),
    #                 octave_range=None, xlabel='Time (seconds)', ylabel='Octaves', title='PianoRoll')
    _create_pianoroll(data =df_d, midi_min=21, midi_max=108)
    #
    #
    # # df_d.to_csv("/home/chris/Documents/Workspace/weimar/example_files/test_measure_diff.csv")
    # # print("This is printed in PANDAS dataframe - Which can be converted to desired format such .csv, numpy array etc.")




if __name__=="__main__":

    main()
"""

start_1 measure_1 start_2_3 measure_2_3

   Start  Duration  Pitch   volume Instrument Event     Measure_Same Measure_p2    
0  0.000     1.000   68.0  0.70866      Piano Chord         1           1
1  0.000     1.000   76.0  0.70866      Piano Note          1           1
2  0.000     1.000   NA    NA           Piano Rest          1           2
2  0.000     1.000   NA    NA           Piano Rest          2           2
2  0.000     1.000   NA    NA           Piano Rest          2           2


N harmonic (exchange) : note can be represented in differnet ways 
was this note, chord or rest,



-------------------------------------------------


Things that are needed:
1. Measure information - done
2. Micro intervals .. Native music - oriental music  
3. Additional col for syml (C, B-..) - because of inharmonic struc. 


Coucple of Exampes: 
1. N grams 
2. SimILARIY APEIACG


ideas fo visulaizaiton: 
1. plot changing measure
2. Which parts are similar and not similar 



Next StepS:
1. Gothrough wiki and create visualization- 
get num of paths
2. Add part as col



quint

4 8ths has 2 quater 
1 quater duration of 1

| ooooo - | - tuplet
| oooo  - | - is it 8ths??  

# For tomorrow
* show the df 


https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel

Piano roll 

"""



