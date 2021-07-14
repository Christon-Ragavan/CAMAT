"""
Working on voices and parts

"""

import sys

sys.path.append('/Users/chris/DocumentLocal/workspace')
from hfm.scripts_in_progress.examples.hfm_database_search import run_search
import os
import re
# from musicxml_parser import scoreToPianoroll
# import musicxml_parser.totalLengthHandler
import tempfile
import xml.etree.ElementTree as ET

import music21 as m21
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests

from music21 import *

print("Pandas Version", pd.__version__)
print("MUSIC21 Version", m21.__version__)

pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 1000000)
pd.set_option('display.width', 1000000)

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
            print(f"... time signature - {b[0]}/{b[1]} -- curr_measure_duration : {self.curr_measure_duration}")

            self.measure_duration_dict.update(
                {str(self.curr_measure_num) + '_' + str(self.curr_part_id): self.curr_measure_duration})

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
                if int(m.attrib['number']) == 1:
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
                            is_alter = p.find('alter')
                            if p.tag == 'pitch':
                                for ppp in p:
                                    if ppp.tag == 'step':
                                        if is_alter != None:
                                            print(is_alter.text, "..t..")
                                            s = ppp.text + is_alter.text
                                            self.step.append(s)

                                        else:
                                            self.step.append(ppp.text)

                                    if ppp.tag == 'octave':
                                        self.octave.append(ppp.text)
                            if p.tag == 'rest':
                                self.step.append(p.tag)
                                self.octave.append(p.tag)

        print(f"self.step {len(self.step)}")
        print(f"self.octave {len(self.octave)}")
        print(f"self.tie {len(self.tie)}")
        print(f"self.duration {len(self.duration)}")
        print(f"self.chord_tags {len(self.chord_tags)}")
        print(f"self.voice_num {len(self.voice_num)}")
        print("curr_measure_num", self.curr_measure_num)
        print("measure_duration_list", len(self.measure_duration_list))
        assert len(self.step) == len(self.octave)
        assert len(self.step) == len(self.tie)
        assert len(self.duration) == len(self.tie)
        assert len(self.duration) == len(self.chord_tags)
        # assert len(self.duration) == len(self.voice_num)
        assert len(self.duration) == len(self.glb_part_id_list)
        # assert self.curr_measure_num== len(self.measure_duration_list)

        notes = []
        for i in range(len(self.step)):
            n = str(self.step[i]) + str(self.octave[i])
            if n == "restrest":
                n = "rest"
            notes.append(n)

        if True:
            print(f"self.note_counter_list      :{len(self.note_counter_list)}")
            print(f"self.duration               :{len(self.duration)}")
            print(f"self.step                   :{len(self.step)}")
            print(f"self.octave                 :{len(self.octave)}")
            print(f"self.measure_num_per_note_li:{len(self.measure_num_per_note_list)}")
            print(f"self.voice_num              :{len(self.voice_num)}")
            print(f"self.glb_part_id_list       :{len(self.glb_part_id_list)}")
            print(f"self.chord_tags             :{len(self.chord_tags)}")
            print(f"self.tie                    :{len(self.tie)}")

        try:

            df_data = pd.DataFrame(np.array(
                [self.note_counter_list, self.duration, self.step, self.octave,
                 self.measure_num_per_note_list, self.voice_num, self.glb_part_id_list,
                 self.chord_tags,
                 self.tie]).T,
                                   columns=["#Note_Debug", "Duration", 'Pitch', 'Octave', 'Measure', 'Voice', 'PartID',
                                            'Chord Tags',
                                            'Tie Type'])
        except:
            df_data = pd.DataFrame(np.array(
                [self.note_counter_list,
                 self.duration,
                 self.step,
                 self.octave,
                 self.measure_num_per_note_list,
                 self.chord_tags,
                 self.tie,
                 self.glb_part_id_list]).T,
                                   columns=["#Note_Debug",
                                            "Duration", 'Pitch', 'Octave', 'Measure',
                                            'Chord Tags',
                                            'Tie Type', 'PartID'])

        return df_data

    def _tie_append_wrapper(self, df, idx, sel_df):
        self.df_data_tie = self.df_data_tie.append(sel_df, ignore_index=True)

    def compute_tie_duration(self, df):

        t_len = len(df)

        for i in range(t_len):
            curr_tie_typ = df['Tie Type'][i]
            try:
                curr_cont = df['Pitch'][i] + df['Octave'][i] + df['Voice'][i]
            except:
                curr_cont = df['Pitch'][i] + df['Octave'][i] + df['PartID'][i]

            if curr_tie_typ == "none":
                sel_df = df.iloc[[i]]
                self._tie_append_wrapper(df, i, sel_df)

            elif curr_tie_typ == 'start':
                l_df_start = df.iloc[[i]].copy()

                l_duration = float(df['Duration'][i])
                for ii in range(i + 1, t_len):

                    ck_type = df['Tie Type'][ii]
                    try:
                        ck_curr_cont = df['Pitch'][ii] + df['Octave'][ii] + df['Voice'][ii]
                    except:
                        ck_curr_cont = df['Pitch'][ii] + df['Octave'][ii] + df['PartID'][ii]

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


def _compute_idx_new_measure_for_multi_parts(measure_num_list):
    idx_new_measure_offsets = []
    m_t = None
    for i, m in enumerate(measure_num_list):
        if i == 0:
            m_t = m
        else:
            if m_t == m:
                continue
            else:
                m_t = m
        idx_new_measure_offsets.append(m_t)
    return idx_new_measure_offsets

def _measure_offset_sparse(measure_num_list, measure_offset, idx_new_measure_offsets):
    measure_offset_sum =[np.sum(measure_offset[:i]) for i in range(1, len(measure_offset)+1)]
    n_sparce_m = []
    n_measure_change_idx =[]
    c=0
    for i, m in enumerate (measure_num_list):
        if m==idx_new_measure_offsets[c]:
            n_measure_change_idx.append(i)
            n_sparce_m.append(measure_offset_sum[c])
            c+=1
        else:
            n_sparce_m.append(0.0)
    return n_sparce_m, n_measure_change_idx, measure_offset_sum
def _merge_measure_duration_and_get_offset(df, measure_offset):
    idx_new_measure_offsets = []

    measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
    voices_list = list(np.squeeze(df['Voice'].to_numpy(dtype=int)))
    duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
    chord_info_list = list(np.squeeze(df['Chord Tags'].to_numpy()))
    n_num_voice = len(list(set(voices_list)))
    n_set_voice = list(set(voices_list))
    offset_list = np.zeros(len(measure_num_list))
    offset_list = list(offset_list)

    idx_new_measure_offsets = _compute_idx_new_measure_for_multi_parts(measure_num_list)  # computing number of measure needed in list
    assert len(idx_new_measure_offsets) == len(
        measure_offset), "Check the lengths len(idx_new_measure_offsets){} !=len(measure_offset) {}".format(
        len(idx_new_measure_offsets), len(measure_offset))

    measure_off_sparse, n_measure_change_idx,measure_offset_sum = _measure_offset_sparse(measure_num_list, measure_offset, idx_new_measure_offsets)

    nn_offset_list = []
    nn_voice_list = []
    nn_measure_list = []

    voice_track_container = [[]for i in range(n_num_voice)]

    c_off = 0
    curr_measure_offset = measure_off_sparse[0]

    for i, c in enumerate(chord_info_list):
        if i in n_measure_change_idx:
            curr_measure_offset = measure_off_sparse[i]
            voice_track_container = [[]for i in range(n_num_voice)]

        if i == 0:
            c_off = 0
            v_trk = voices_list[i]
            m_trk = measure_num_list[i]
            c_dur = duration_list[i]

            voice_track_container[v_trk-1].append([c_off, c_dur])
        else:
            v_trk = voices_list[i]
            m_trk = measure_num_list[i]
            c_dur = duration_list[i]

            voice_track_container[v_trk-1].append([c_off, c_dur])

            """
            if nn_measure_list[i-1] != m_trk:
                print("Changed Measure:", m_trk, nn_measure_list[i-1])
                
                if nn_voice_list[i-1] != v_trk:
                    print("Changed Voice:", v_trk, nn_voice_list[i - 1])
                    if c=='chord':
                        print("Chord Found:")
                    else:
                        print("Chord Not Found")
                else:
                    print("Same Voice:", v_trk, nn_voice_list[i - 1])
                    if c=='chord':
                        print("Chord Found:")
                    else:
                        print("Chord Not Found")
            else:
                print("Same Measure:", m_trk, nn_measure_list[i-1])
                if nn_voice_list[i-1] != v_trk:
                    print("Changed Voice:", v_trk, nn_voice_list[i - 1])
                    if c=='chord':
                        print("Chord Found:")
                    else:
                        print("Chord Not Found")
                else:
                    print("Same Voice:", v_trk, nn_voice_list[i - 1])
                    if c=='chord':
                        print("Chord Found:")
                    else:
                        print("Chord Not Found")

            """
            if c == 'chord':
                c_off = nn_offset_list[i - 1]
                if nn_voice_list[i - 1] != v_trk:
                    if len(voice_track_container[v_trk-1])==1:
                        c_off = curr_measure_offset
                    else:
                        # prob
                        c_off = nn_offset_list[i - 1]
                else:
                    pass
            else:
                c_off = nn_offset_list[i - 1] + duration_list[i - 1]
                print(len(voice_track_container[v_trk-1]))
                if nn_voice_list[i-1]!= v_trk:
                    if len(voice_track_container[v_trk-1])==1:
                        c_off = curr_measure_offset
                    else:
                        # prob
                        c_off = nn_offset_list[i - 1]
                else:
                    pass
                    # prob
                    #c_off = nn_offset_list[i - 1] + duration_list[i - 1]
        print(voice_track_container)
        print("off:", c_off, "voc", v_trk, "mea", m_trk)

        nn_offset_list.append(c_off)
        nn_voice_list.append(v_trk)
        nn_measure_list.append(m_trk)

    print("\n\n\n------------")
    print(voice_track_container)
    print("\n\n\n------------")

    assert len(nn_offset_list) == len(df), "Check the lengths len(nn_offset_list){} !=len(df) {}".format(
        len(nn_offset_list), len(df))

    df.insert(loc=1, column='Offset', value=nn_offset_list)
    #df.insert(loc=3, column='voice_track', value=nn_voice_list)
    print(df)
    return df


def _remove_unwanted_cols(df):
    df.pop("Chord Tags")
    df.pop("Tie Type")
    return df


def _scrape_database(search_keywords):
    database_csv_path = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/database/'
    df_s = run_search(search_keywords=search_keywords,
                      extract_database=False,
                      apply_precise_keyword_search=True,
                      save_extracted_database_path=database_csv_path,
                      save_search_output_path='search_output.csv')

    return df_s


def _download_xml_file(xml_link):
    save_at = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files/web_xml/'

    saved_links = []
    for link in xml_link:
        file_name = link.split('/')[-1]
        print("Downloading file:%s" % file_name)
        response = requests.get(link)
        s = save_at + file_name
        with open(s, 'wb') as file:
            file.write(response.content)
        saved_links.append(s)
        print("%s downloaded!\n" % file_name)
    return saved_links


def _get_file(search_keywords):
    base_dir = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files'
    if True:
        # path = base_dir + "/test_case_xml_parser_example.xml"
        # path = base_dir + "/test_case_tied_note.xml"
        # path = base_dir + "/ultimate_tie_test.xml"
        # path = base_dir + "/ultimate_tie_test2.xml"
        # path = base_dir + "/sacred_xml.xml"
        # path = base_dir + "/test_case_ives1.xml"
        # path = base_dir + "/ultimate_tie_test3.xml"
        path = base_dir + "/ultimate_tie_test4.xml"
        path = base_dir + "/ultimate_tie_test5.xml"
        # path = base_dir + "/Untitled6.xml"
        # path = base_dir + "/stupid2.xml"
        # path = base_dir + "/BrumAn_Bru1011_COM_3-6_MissaProde_002_01134.xml"
        # path = base_dir + "/test_case_Castuski_1_time_sig_measure.xml"

    else:
        df_s = _scrape_database(search_keywords)
        urls = df_s['url'].to_list()
        urls = urls[:1]
        paths = _download_xml_file(urls)
        path = paths[0]

    assert os.path.isfile(path), "File not found {}".format(path)
    print("----------------PARSING--------------------")
    print(path)
    b = converter.parse(path)
    b.show('text')
    return path


if __name__ == "__main__":
    """
    Stabel Extractor! TO CHECEK duration 

    """

    search_keywords = {'Composer': ['josquin'],
                       'Movement Number': None,
                       'Title': ['missa', ],
                       'Key': None,
                       'Life Time Year': None,
                       'Life Time Range': None,
                       'Year Range': None}

    path = _get_file(search_keywords)
    xml_tools = XMLToolBox(file_path=path)
    df_data = xml_tools.strip_xml()

    print("INITIAL XML EXTRACT")
    print(np.shape(df_data))
    print(df_data)
    df_data_chord = _merge_measure_duration_and_get_offset(df_data, xml_tools.measure_duration_list)
    print("After combining Chord Offser")
    print(np.shape(df_data_chord))
    # print(df_data_chord)

    df_data_chord_tied = xml_tools.compute_tie_duration(df_data)
    print("After Parsing Tied info")
    print(np.shape(df_data_chord_tied))

    #print(df_data_chord_tied)

    print("FINAL")
    df_f = _remove_unwanted_cols(df_data_chord_tied)
    #print(df_f)
    #print(np.shape(df_f))

    # print(df_f)

""" 


Part 1
    Measure 1 
        Voice 1
            note
            chord
            tie
        Voice 2
Part 2

- dynamic mentioned once



TODO: 
+ basic structure understanding  

Meet at 15:00
"""