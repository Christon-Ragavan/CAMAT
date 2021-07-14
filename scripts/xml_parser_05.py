import numpy as np
import pandas as pd
from music21 import *
import music21 as m21
import os
import xml.sax
# from musicxml_parser import scoreToPianoroll
# import musicxml_parser.totalLengthHandler
from musicxml_parser.scoreToPianoroll import scoreToPianoroll
import matplotlib.pyplot as plt
import tempfile
import re
import xml.etree.ElementTree as ET

print("Pandas Version", pd.__version__)
print("MUSIC21 Version", m21.__version__)
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.width', 100000)

us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()
us['musescoreDirectPNGPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'
us['musicxmlPath'] = r'/Applications/MuseScore 3.app/Contents/MacOS/mscore'

"""
      <note default-x="84">
        <pitch>
          <step>A</step>
          <octave>4</octave>
        </pitch>
        <duration>8</duration>
        <voice>1</voice>
        <type>whole</type>
      </note>
    4*quater/4
    3/4
    <time> 
          <beats>4</beats>
          <beat-type>4</beat-type>
    </time>
"""

mapping_step_midi = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11
}

mapping_dyn_number = {
    # Value drawn from http://www.wikiwand.com/en/Dynamics_%28music%29
    'ppp': 0.125,
    'pp': 0.258,
    'p': 0.383,
    'mp': 0.5,
    'mf': 0.625,
    'f': 0.75,
    'ff': 0.875,
    'fff': 0.984
}


class XMLToolBox:
    def __init__(self, file_path):
        self.file_path = self.pre_process_file(file_path)
        score = ET.parse(self.file_path)
        self.root = score.getroot()
        self.num_voices = self._get_num_voices()
        self.curr_measure_divisions = 0
        self.time_offset = []
        self.measure_offset = []
        self.curr_measure_offset = None
        self.note_counter = 0
        self.note_counter_list = []
        self.df_data_tie = pd.DataFrame()

    def pre_process_file(self, file_path):
        temp_file = tempfile.NamedTemporaryFile('w', suffix='.xml', prefix='tmp', delete=False, encoding='utf-8')

        # Remove the doctype line
        with open(file_path, 'r', encoding="utf-8") as fread:
            for line in fread:
                if not re.search(r'<!DOCTYPE', line):
                    temp_file.write(line)
        temp_file_path = temp_file.name
        temp_file.close()
        return temp_file_path

    def _get_num_voices(self):
        n_voice = [v.text for v in self.root.iter('voice')]

        return len(list(set(n_voice)))

    def _find_ties(self, itt, containder_list):
        info_t = itt.findall('tie')
        if len(info_t) == 0:
            containder_list.append("None")
        elif len(info_t) == 1:
            containder_list.append(info_t[0].attrib['type'])

        elif len(info_t) == 2:
            ck = np.sort([i.attrib['type'] for i in info_t])
            if ck[0] == 'start' and ck[1] == 'stop':
                containder_list.append("continue")
        else:
            ck = [i.attrib['type'] for i in info_t]
            raise Exception(f"ERROR: Tie Type Not Understood {ck}")

        if False:
            # if True:
            info = itt.find('tie')

            if info != None:
                containder_list.append(info.attrib['type'])
            else:
                containder_list.append("None")

    def _find_chords(self, itt, containder_list):
        chord_i = itt.find('chord')
        if chord_i != None:
            self.set_chord = True
            containder_list.append('chord')
        else:
            self.set_chord = False
            containder_list.append('None')

    def _set_curr_measure_offset(self, itt):
        time_i = itt.find('time')

        if time_i != None:
            b = [None, None]

            for t in time_i:
                if t.tag == 'beats':
                    b[0] = float(t.text)
                if t.tag == 'beat-type':
                    b[1] = float(t.text)

            d = float(b[0] / b[1])
            self.curr_measure_offset = float(4 * d)
            print(f"... time signature - {b[0]}/{b[1]} -- offset : {self.curr_measure_offset}")
        else:
            pass

    def strip_xml(self):
        c = 0
        step = []
        octave = []
        tie = []
        duration = []
        self.voice_num = []
        chord_tags = []
        pitch_notes_v = [[] for i in range(self.num_voices)]
        octave_v = [[] for i in range(self.num_voices)]
        self.set_chord = False
        self.measure_number_list = []
        self.measure_num_per_note_list = []
        self.curr_measure_num = 0
        self.voices_nd_list = [[] for i in range(self.num_voices)]
        self.voices = []
        for part in self.root.iter('part'):
            print(f" ------------- part {part.tag} {part.attrib['id']} -------------")
            for m in self.root.iter('measure'):
                print(f"measure {m.tag} {m.attrib['number']} ")
                self.curr_measure_num = int(m.attrib['number'])
                self.measure_number_list.append(self.curr_measure_num)

                for m_itt in m:  # itterate over measure
                    if m_itt.tag == 'attributes':
                        division_i = m_itt.find('divisions')
                        self._set_curr_measure_offset(itt=m_itt)

                        if division_i != None:
                            self.curr_measure_divisions = float(division_i.text)

                    if m_itt.tag == 'note':
                        self.note_counter += 1
                        self.note_counter_list.append(self.note_counter)
                        self.measure_num_per_note_list.append(self.curr_measure_num)
                        # print(f" # Note num {self.note_counter} -- {self.curr_measure_offset}")
                        self._find_ties(itt=m_itt, containder_list=tie)
                        self._find_chords(itt=m_itt, containder_list=chord_tags)

                        for p in m_itt:  # itterate over pitches
                            c += 1
                            if p.tag == 'voice':
                                self.voice_num.append(p.text)

                            if p.tag == 'duration':
                                duration.append(float(float(p.text) / self.curr_measure_divisions))

                            if p.tag == 'pitch':
                                for ppp in p:
                                    if ppp.tag == 'step':
                                        step.append(ppp.text)

                                    if ppp.tag == 'octave':
                                        octave.append(ppp.text)
                            if p.tag == 'rest':
                                step.append(p.tag)
                                octave.append(p.tag)
        print(f"step {len(step)}")
        print(f"octave {len(octave)}")
        print(f"tie {len(tie)}")
        print(f"duration {len(duration)}")
        print(f"chord_tags {len(chord_tags)}")
        print(f"self.voice_num {len(self.voice_num)}")
        assert len(step) == len(octave)
        assert len(step) == len(tie)
        assert len(duration) == len(tie)
        assert len(duration) == len(chord_tags)
        assert len(duration) == len(self.voice_num)

        notes = []

        for i in range(len(step)):
            n = str(step[i]) + str(octave[i])
            if n == "restrest":
                n = "rest"
            notes.append(n)
        df_data = pd.DataFrame(np.array(
            [self.note_counter_list, duration, step, octave, self.measure_num_per_note_list, self.voice_num, chord_tags,
             tie]).T,
                                   columns=["#Note", "Duration", 'Pitch', 'Octave', 'Measure', 'Voice', 'Chord Tags',
                                            'Tie Type'])



        return df_data

    def _tie_append_wrapper(self, df, idx, sel_df):
        self.df_data_tie = self.df_data_tie.append(sel_df, ignore_index=True)

    def compute_tie_duration(self, df):
        print("compute_tie_duration")
        print(df)
        print(df["Pitch"][1])
        t_len = len(df)

        for i in range(t_len):
            curr_tie_typ = df['Tie Type'][i]
            curr_cont = df['Pitch'][i]+df['Octave'][i]+df['Voice'][i]

            print(".. ",curr_tie_typ)
            if curr_tie_typ == 'None':
                sel_df = df.iloc[[i]]
                self._tie_append_wrapper(df, i, sel_df)

            elif curr_tie_typ == 'start':
                print(f".. .. .. .. .. .. Range {i} /{t_len}")
                l_df_start = df.iloc[[i]].copy()


                l_duration = float(df['Duration'][i])
                for ii in range(i+1, t_len):

                    ck_type =  df['Tie Type'][ii]
                    ck_curr_cont = df['Pitch'][ii] + df['Octave'][ii] + df['Voice'][ii]


                    if curr_cont == ck_curr_cont and ck_type == 'continue':
                        l_duration += float(df['Duration'][ii])
                        df.iloc[ii, df.columns.get_loc('Tie Type')] = 'SKIPED_continue'



                    if curr_cont == ck_curr_cont and ck_type == 'stop':
                        l_duration += float(df['Duration'][ii])
                        # l_df_start[0,'Duration'] = l_duration
                        l_df_start.iloc[0, l_df_start.columns.get_loc('Duration')] = l_duration
                        # l_df_start.loc[0, 'Duration'] = l_duration
                        self._tie_append_wrapper(df, ii, l_df_start)
                        df.iloc[ii, df.columns.get_loc('Tie Type')] = 'SKIPED_stop'


                        break

                pass




        print("\n ###### VISUALIZE WHAT ROWs SKIPPED ######\n")
        print(df)
        print("\n ###### VISUALIZE Computed Ties ######\n")

        print(self.df_data_tie)


def _get_file():
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_xml_parser_example.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_tied_note.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test2.xml"
    path = "/hfm/scripts_in_progress/example_files/ultimate_tie_test3.xml"

    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/sacred_xml.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_ives1.xml"
    path = "/hfm/scripts_in_progress/example_files/stupid2.xml"
    b = converter.parse(path)
    b.show('text')
    return path


if __name__ == "__main__":
    """
    Stabel Extractor! TO CHECEK duration 

    """

    path = _get_file()

    xml_tools = XMLToolBox(file_path=path)
    df_data = xml_tools.strip_xml()
    df_data_tied = xml_tools.compute_tie_duration(df_data)


    """
    ref_cont = list(np.squeeze(df_data.iloc[[0]].to_numpy()))
    ref_list = ['#Note', 'Duration', 'Pitch', 'Octave', 'Measure', 'Voice', 'Chord Tags', 'Tie Type']
    d_ne = dict(zip( ref_list, ref_cont))
    print(d_ne)
    df = pd.DataFrame()

    df = df.append(d_ne,  ignore_index=True)
    print("--------kbkjbkb")
    print(df)"""