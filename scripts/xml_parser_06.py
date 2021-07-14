import os
import re
# from musicxml_parser import scoreToPianoroll
# import musicxml_parser.totalLengthHandler
import tempfile
import xml.etree.ElementTree as ET

import music21 as m21
import numpy as np
import pandas as pd
from music21 import *

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
        self.part_id_list = []
        self.glb_part_id_list = []
        self.curr_part_id = None
        self.curr_measure_offset = None
        self.note_counter = 0
        self.note_counter_list = []
        self.df_data_tie = pd.DataFrame()
        self.measure_duration_dict = dict()
        self.measure_duration_list = []


        self.step = []
        self.octave = []
        self.tie = []
        self.duration = []
        self.voice_num = []
        self.chord_tags = []

        self.set_chord = False
        self.measure_number_list = []
        self.measure_num_per_note_list = []
        self.curr_measure_num = 0
        self.voices_nd_list = [[] for i in range(self.num_voices)]
        self.voices = []
        self.curr_measure_duration = 0.0

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

    def _find_ties(self, itt):
        info_t = itt.findall('tie')
        if len(info_t) == 0:
            self.tie.append("none")
        elif len(info_t) == 1:
            self.tie.append(info_t[0].attrib['type'])

        elif len(info_t) == 2:
            ck = np.sort([i.attrib['type'] for i in info_t])
            if ck[0] == 'start' and ck[1] == 'stop':
                self.tie.append("continue")
        else:
            ck = [i.attrib['type'] for i in info_t]
            raise Exception(f"ERROR: Tie Type Not Understood {ck}")

        if False:
            # if True:
            info = itt.find('tie')

            if info != None:
                self.tie.append(info.attrib['type'])
            else:
                self.tie.append("none")

    def _find_chords(self, itt):
        chord_i = itt.find('chord')
        if chord_i != None:
            self.set_chord = True
            self.chord_tags.append('chord')
        else:
            self.set_chord = False
            self.chord_tags.append("none")

    def _set_curr_measure_duration(self, itt):
        time_i = itt.find('time')

        if time_i != None:
            b = [None, None]

            for t in time_i:
                if t.tag == 'beats':
                    b[0] = float(t.text)
                if t.tag == 'beat-type':
                    b[1] = float(t.text)

            d = float(b[0] / b[1])
            self.curr_measure_duration = float(4 * d)
            print(f"... time signature - {b[0]}/{b[1]} -- offset : {self.curr_measure_duration}")

            self.measure_duration_dict.update({str(self.curr_measure_num)+'_'+str(self.curr_part_id): self.curr_measure_duration})

    def strip_xml(self):
        c = 0

        for part in self.root.iter('part'):
            print(f" ------------- part {part.tag} {part.attrib['id']} -------------")
            self.part_id_list.append(part.attrib['id'])
            self.curr_part_id = part.attrib['id']

            for m in self.root.iter('measure'):
                print(f"measure {m.tag} {m.attrib['number']} ")
                self.curr_measure_num = int(m.attrib['number'])
                self.measure_number_list.append(self.curr_measure_num)
                if int(m.attrib['number']) ==1:
                    self.measure_duration_list.append(0.0)
                else:
                    self.measure_duration_list.append(self.curr_measure_duration)
                for m_itt in m:  # itterate over measure
                    if m_itt.tag == 'attributes':
                        division_i = m_itt.find('divisions')

                        self._set_curr_measure_duration(itt=m_itt)

                        if division_i != None:
                            self.curr_measure_divisions = float(division_i.text)


                    if m_itt.tag == 'note':
                        self.note_counter += 1
                        self.glb_part_id_list.append(self.curr_part_id)
                        self.note_counter_list.append(self.note_counter)
                        self.measure_num_per_note_list.append(self.curr_measure_num)
                        # print(f" # Note num {self.note_counter} -- {self.curr_measure_duration}")
                        self._find_ties(itt=m_itt)
                        self._find_chords(itt=m_itt)

                        for p in m_itt:  # itterate over pitches
                            c += 1
                            if p.tag == 'voice':
                                self.voice_num.append(p.text)

                            if p.tag == 'duration':
                                self.duration.append(float(float(p.text) / self.curr_measure_divisions))

                            if p.tag == 'pitch':
                                for ppp in p:
                                    if ppp.tag == 'step':
                                        self.step.append(ppp.text)

                                    if ppp.tag == 'octave':
                                        self.octave.append(ppp.text)
                            if p.tag == 'rest':
                                self.step.append(p.tag)
                                self.octave.append(p.tag)

        #print(f"self.step {len(self.step)}")
        #print(f"self.octave {len(self.octave)}")
        #print(f"self.tie {len(self.tie)}")
        #print(f"self.duration {len(self.duration)}")
        #print(f"self.chord_tags {len(self.chord_tags)}")
        #print(f"self.voice_num {len(self.voice_num)}")
        #print("measure_duration_list",len(self.measure_duration_list))
        assert len(self.step) == len(self.octave)
        assert len(self.step) == len(self.tie)
        assert len(self.duration) == len(self.tie)
        assert len(self.duration) == len(self.chord_tags)
        assert len(self.duration) == len(self.voice_num)
        assert len(self.duration) == len(self.glb_part_id_list)
        assert self.curr_measure_num== len(self.measure_duration_list)

        notes = []
        for i in range(len(self.step)):
            n = str(self.step[i]) + str(self.octave[i])
            if n == "restrest":
                n = "rest"
            notes.append(n)


        df_data = pd.DataFrame(np.array(
            [self.note_counter_list, self.duration, self.step, self.octave, self.measure_num_per_note_list, self.voice_num,self.glb_part_id_list , self.chord_tags,
             self.tie]).T,
                                   columns=["#Note_Debug", "Duration", 'Pitch', 'Octave', 'Measure', 'Voice', 'PartID','Chord Tags',
                                            'Tie Type'])



        return df_data

    def _tie_append_wrapper(self, df, idx, sel_df):
        self.df_data_tie = self.df_data_tie.append(sel_df, ignore_index=True)

    def compute_tie_duration(self, df):

        t_len = len(df)

        for i in range(t_len):
            curr_tie_typ = df['Tie Type'][i]
            curr_cont = df['Pitch'][i]+df['Octave'][i]+df['Voice'][i]

            if curr_tie_typ == "none":
                sel_df = df.iloc[[i]]
                self._tie_append_wrapper(df, i, sel_df)

            elif curr_tie_typ == 'start':
                l_df_start = df.iloc[[i]].copy()


                l_duration = float(df['Duration'][i])
                for ii in range(i+1, t_len):

                    ck_type =  df['Tie Type'][ii]
                    ck_curr_cont = df['Pitch'][ii] + df['Octave'][ii] + df['Voice'][ii]


                    if curr_cont == ck_curr_cont and ck_type == 'continue':
                        l_duration += float(df['Duration'][ii])
                        df.iloc[ii, df.columns.get_loc('Tie Type')] = 'SKIPPED_continue'



                    if curr_cont == ck_curr_cont and ck_type == 'stop':
                        l_duration += float(df['Duration'][ii])
                        # l_df_start[0,'Duration'] = l_duration
                        l_df_start.iloc[0, l_df_start.columns.get_loc('Duration')] = l_duration
                        # l_df_start.loc[0, 'Duration'] = l_duration
                        self._tie_append_wrapper(df, ii, l_df_start)
                        df.iloc[ii, df.columns.get_loc('Tie Type')] = 'SKIPPED_stop'
                        break
                pass
        return self.df_data_tie


def _merge_measure_duration_and_get_offset(df, measure_offset):

    idx_offsets = []

    measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
    duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
    chord_info_list = list(np.squeeze(df['Chord Tags'].to_numpy()))

    offset_list = np.zeros(len(measure_num_list))
    offset_list = list(offset_list)
    num_m = np.max(measure_num_list)
    print(measure_offset, np.sum(measure_offset))
    for i in range(num_m):
        try:
            idx_offsets.append(measure_num_list.index(i+1))
        except:
            measure_offset.pop(i)

            continue
    assert len(idx_offsets) == len(measure_offset)

    for idx, i_off in enumerate(idx_offsets):
        offset_list[i_off] = measure_offset[idx]
    assert len(offset_list) == len(measure_num_list)

    nn_offset_list = []
    c_off = 0
    for i, c in enumerate(chord_info_list):
        if i == 0:
            c_off = 0
        else:
            if c == 'chord':
                temp_ch = []
                c_off = nn_offset_list[i - 1]
                if chord_info_list[i + 1] == "none":
                    pass
            else:
                c_off = nn_offset_list[i - 1] + duration_list[i - 1]

        nn_offset_list.append(c_off)
    assert len(nn_offset_list)==len(df)

    try:
        df.insert(loc=1, column='Offset1', value=nn_offset_list)
    except:
        df.insert(loc=1, column='Offset2', value=nn_offset_list)
    return df

def _get_file():
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_xml_parser_example.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_tied_note.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test2.xml"

    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/sacred_xml.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_ives1.xml"

    path = "/hfm/scripts_in_progress/example_files/ultimate_tie_test3.xml"
    #path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/stupid2.xml"
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
    print("INITIAL XML EXTRACT")
    print(df_data)

    df_test = _merge_measure_duration_and_get_offset(df_data,xml_tools.measure_duration_list )
    print("INITIAL XML test")

    print(df_test)

    df_data_tied = xml_tools.compute_tie_duration(df_data)
    print("After Parsing Tied info")
    print(df_data_tied)

    df_data_tied_chord = _merge_measure_duration_and_get_offset(df_data_tied,xml_tools.measure_duration_list )

    print("After Parsing Tied info and computing Chord offset durtion ")
    print(df_data_tied_chord)


    print(set(df_data_tied_chord['Offset1']))
