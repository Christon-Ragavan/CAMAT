"""
Working on voices and parts

"""

import sys

sys.path.append('/Users/chris/DocumentLocal/workspace')
from hfm.scripts_in_progress.examples.hfm_database_search import run_search
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from progressbar import progressbar as pbar
import music21 as m21
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
from hfm.scripts_in_progress.examples.pianoroll_matplotlib import plot_pianoroll
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
        self.measure_offset_list = []

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
        self.curr_measure_offset = 0.0

        self.dict_12_pc = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                           'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        self.rev_dict_12_pc = {v: k for k, v in self.dict_12_pc.items()}


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
            self.curr_measure_offset = float(4 * d)
            # print(f"... time signature - {b[0]}/{b[1]} -- curr_measure_offset : {self.curr_measure_offset}")

            self.measure_duration_dict.update(
                {str(self.curr_measure_num) + '_' + str(self.curr_part_id): self.curr_measure_offset})
    def _strip_part_information(self):
        # print("_strip_part_information")
        for p_meta in self.root.iter('part-list'):
            # print(f" --- part-list {p_meta.tag}")
            pass





    def strip_xml(self):
        self._strip_part_information()

        c = 0
        self.part_id_counter = 0

        for part in self.root.iter('part'):
            self.part_id_counter +=1
            # print(f" ------------- part {part.tag} {part.attrib['id']} {self.part_id_counter}-------------")
            self.part_id_list.append(part.attrib['id'])
            self.curr_part_id = self.part_id_counter
            # self.curr_part_id = part.attrib['id']

            for m in part:
                # print(f"measure {m.tag} {m.attrib['number']} ")
                self.curr_measure_num = int(m.attrib['number'])
                self.measure_number_list.append(self.curr_measure_num)
                if int(m.attrib['number']) == 1:
                    self.measure_offset_list.append(0.0)
                else:
                    self.measure_offset_list.append(self.curr_measure_offset)
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
                        # print(f" # Note num {self.note_counter} -- {self.curr_measure_offset}")
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
                                            s = ppp.text + is_alter.text
                                            self.step.append(s)

                                        else:
                                            self.step.append(ppp.text)

                                    if ppp.tag == 'octave':
                                        self.octave.append(ppp.text)
                            if p.tag == 'rest':
                                self.step.append(p.tag)
                                self.octave.append(p.tag)

        #print(f"self.step               :{len(self.step)}")
        #print(f"self.octave             :{len(self.octave)}")
        #print(f"self.tie                :{len(self.tie)}")
        #print(f"self.duration           :{len(self.duration)}")
        #print(f"self.chord_tags         :{len(self.chord_tags)}")
        #print(f"self.voice_num          :{len(self.voice_num)}")
        #print("curr_measure_num         :", self.curr_measure_num)
        #print("measure_offset_list      :", len(self.measure_offset_list))
        assert len(self.step) == len(self.octave)
        assert len(self.step) == len(self.tie)
        assert len(self.duration) == len(self.tie)
        assert len(self.duration) == len(self.chord_tags)
        # assert len(self.duration) == len(self.voice_num)
        assert len(self.duration) == len(self.glb_part_id_list)
        # assert self.curr_measure_num== len(self.measure_offset_list)

        notes = []
        for i in range(len(self.step)):
            n = str(self.step[i]) + str(self.octave[i])
            if n == "restrest":
                n = "rest"
            notes.append(n)


        if False:
            print(f"self.note_counter_list      :{len(self.note_counter_list)}")
            print(f"self.duration               :{len(self.duration)}")
            print(f"self.step                   :{len(self.step)}")
            print(f"self.octave                 :{len(self.octave)}")
            print(f"self.measure_num_per_note_li:{len(self.measure_num_per_note_list)}")
            print(f"self.voice_num              :{len(self.voice_num)}")
            print(f"self.glb_part_id_list       :{len(self.glb_part_id_list)}")
            print(f"self.chord_tags             :{len(self.chord_tags)}")
            print(f"self.tie                    :{len(self.tie)}")


        if len(self.voice_num)==0:
            self.voice_num = [1] * len(self.duration)


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
                 self.voice_num,
                 self.chord_tags,
                 self.tie,
                 self.glb_part_id_list]).T,
                                   columns=["#Note_Debug",
                                            "Duration", 'Pitch', 'Octave', 'Measure', 'Voice',
                                            'Chord Tags',
                                            'Tie Type', 'PartID'])

        return df_data

    def _tie_append_wrapper(self, df, idx, sel_df):
        self.df_data_tie = self.df_data_tie.append(sel_df, ignore_index=True)


    def _pitch_to_midi(self, octave, pitch):
        if octave == 'rest':
            m = None
        else:
            octave = int(octave)
            try:
                pitch_class = self.dict_12_pc[pitch]
            except:
                print("error in _pitch_to_midi()", pitch)
            o = (octave + 1) * 12
            m = int(pitch_class + o)
        return m
    def _midi_list(self):
        pass

    def _parse_pitch_ann(self, pitch_list):
        pattern = '(^[a-z|A-Z]{1})(\-?)(\d*$)'
        pl = []
        shifter = []
        for ps in pitch_list:
            if 'rest' in ps:
                pl.append(ps)
                shifter.append(0)
            else:
                match = re.search(pattern, ps)
                if match:
                    if match.group(1):
                        pl.append(match.group((1)))
                    else:
                        raise Exception(f"Error in pitch")
                    if match.group((3)):
                        if match.group((2)):
                            shifter.append(-float(match.group((3))))
                        else:
                            shifter.append(float(match.group((3))))
                    else:
                        shifter.append(0)
                else:
                    pl.append(np.NAN)
                    shifter.append(np.NAN)
        assert len(pitch_list) == len(shifter), f"checking org pitch_list {len(pitch_list)} shifter {len(shifter)}"
        assert len(pl) == len(shifter), f"checking pl {len(pl)} shifter {len(shifter)}"

        return pl, shifter

    def convert_pitch_midi(self, df):
        pitch_list = list(np.squeeze(df['Pitch'].to_numpy(dtype=object)))
        octave_list = list(np.squeeze(df['Octave'].to_numpy(dtype=object)))
        t_pl, t_shifter = self._parse_pitch_ann(pitch_list)


        t_pl = [self._pitch_to_midi(octave_list[i], t_pl[i])for i in range(len(t_pl))]
        self.midi_list = [ t_pl[i]+t_shifter[i] if t_pl[i] !=None  else None for i in range(len(t_pl))]

        df.insert(loc=6, column='MIDI', value=self.midi_list)
        return df



    def compute_tie_duration(self, df):

        t_len = len(df)

        for i in range(t_len):
            curr_tie_typ = df['Tie Type'][i]
            # TODO: write some error handeler
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


    @staticmethod
    def _measure_offset_sparse(measure_num_list, partid_num_list, measure_offset, idx_new_measure_offsets):

        """        print("measure_offset")
                print(measure_offset, len(measure_offset))
                print("measure_num_list")
                print(measure_num_list, len(measure_num_list))
                print("idx_new_measure_offsets")
                print(idx_new_measure_offsets, len(idx_new_measure_offsets))
        """
        assert len(idx_new_measure_offsets) == len(measure_offset)
        measure_offset_sum = []
        n_sparce_m = []
        n_measure_change_idx = []
        c = 0
        for i in range(1, len(measure_offset)+1):
            if i < len(measure_offset):
                measure_offset_sum.append(np.sum(measure_offset[:i]))

                if idx_new_measure_offsets[i] == 1:
                    measure_offset[:i] = [0]*i
            else:
                measure_offset_sum.append(np.sum(measure_offset))

        assert len(measure_offset_sum)==len(measure_offset), f"Error in generating measure offset sum, measure_offset_sum:  {len(measure_offset_sum)}, measure_offset: {len(measure_offset)}"
        for i, m in enumerate(measure_num_list):
            try:
                if m == idx_new_measure_offsets[c]:
                    n_measure_change_idx.append(i)
                    curr_measure = measure_offset_sum[c]
                    n_sparce_m.append(curr_measure)
                    c += 1
                else:
                    n_sparce_m.append(curr_measure)
            except:
                n_sparce_m.append(curr_measure)

                continue

        return n_sparce_m, n_measure_change_idx, measure_offset_sum

    @staticmethod
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


    def compute_measure_offset(self, df):
        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        partid_num_list = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))

        idx_new_measure_offsets = xml_tools._compute_idx_new_measure_for_multi_parts(measure_num_list)  # computing number of measure needed in list
        assert len(idx_new_measure_offsets) == len(self.measure_offset_list), "Check the lengths len(idx_new_measure_offsets){} !=len(measure_offset) {}".format(
            len(idx_new_measure_offsets), len(self.measure_offset_list))

        self.measure_off_sparse, self.n_measure_change_idx, measure_offset_sum = xml_tools._measure_offset_sparse(measure_num_list,partid_num_list,
                                                                                                        self.measure_offset_list,
                                                                                                        idx_new_measure_offsets)
        df.insert(loc=1, column='Offset_ml', value=self.measure_off_sparse)
        return df


    def compute_voice_offset(self, df):
        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        offset_ml_num_list = list(np.squeeze(df['Offset_ml'].to_numpy(dtype=int)))
        voices_list = list(np.squeeze(df['Voice'].to_numpy(dtype=int)))
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['Chord Tags'].to_numpy()))
        part_id_list = list(np.squeeze(df['PartID'].to_numpy()))
        n_num_voice = len(list(set(voices_list)))
        nn_offset_list = []
        nn_voice_list = []
        nn_measure_list = []
        voice_track_container = [[] for i in range(n_num_voice)]

        c_off = 0.0
        curr_measure_offset = 0.0
        measure_C = 0
        for i, v in enumerate(voices_list):
            v_trk = voices_list[i]
            c_ch = chord_info_list[i]
            m_trk = measure_num_list[i]
            pid_trk = part_id_list[i]
            c_dur = duration_list[i]

            if i in self.n_measure_change_idx: # resetting the container at every change in measure
                measure_C+=1
                curr_measure_offset = self.measure_off_sparse[i]
                del  voice_track_container
                voice_track_container = [[] for i in range(n_num_voice)]
                # print("########### Curr Measure ",measure_C," ###############", curr_measure_offset, np.shape(voice_track_container))

            if part_id_list[i - 1] != pid_trk:  # resetting the container at every change in measure
                c_off = 0.0


            if i == 0:
                c_off = 0.0
            else:

                if part_id_list[i - 1] != pid_trk:  # resetting the container at every change in measure
                    c_off = 0.0
                    del voice_track_container
                    voice_track_container = [[] for i in range(n_num_voice)]

                elif voices_list[i - 1] != v:
                    if measure_num_list[i-1]!=m_trk:
                        c_off = curr_measure_offset
                    elif len(voice_track_container[v-1])==0:
                        c_off = curr_measure_offset
                    else:
                        c_off = np.sum(voice_track_container[v-1][-1])
                else:
                    if measure_num_list[i-1]!=m_trk:
                        c_off = curr_measure_offset
                    elif c_ch == 'chord':
                        c_off = nn_offset_list[i - 1]
                    else:
                        c_off = nn_offset_list[i-1]+duration_list[i-1]




            voice_track_container[v - 1].append([c_off, c_dur])
            nn_offset_list.append(c_off)
        df.insert(loc=2, column='Offset', value=nn_offset_list)
        return df


    def compute_chord_offset(self, df):
        nn_offset_list=[]
        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        voices_list = list(np.squeeze(df['Voice'].to_numpy(dtype=int)))
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['Chord Tags'].to_numpy()))
        n_num_voice = len(list(set(voices_list)))
        n_set_voice = list(set(voices_list))
        offset_list = np.zeros(len(measure_num_list))
        offset_list = list(offset_list)
        num_m = np.max(measure_num_list)

        idx_new_measure_offsets = _compute_idx_new_measure_for_multi_parts(
            measure_num_list)  # computing number of measure needed in list

        nn_offset_list = []

        for i, c in enumerate(chord_info_list):
            if i == 0:
                c_off = 0
            else:
                if c == 'chord':
                    c_off = nn_offset_list[i - 1]
                else:
                    c_off = nn_offset_list[i - 1] + duration_list[i - 1]

            nn_offset_list.append(c_off)

        df.insert(loc=3, column='Offset_cl', value=nn_offset_list)
        return df


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




def _remove_unwanted_cols(df):
    df.pop("Chord Tags")
    df.pop("Tie Type")
    df.pop("Offset_ml")
    return df


def _scrape_database(search_keywords,extract_extire_database):
    database_csv_path = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/database/'
    df_s = run_search(search_keywords=search_keywords,
                      extract_database=False,
                      apply_precise_keyword_search=True,
                      extract_extire_database=extract_extire_database,
                      save_extracted_database_path=database_csv_path,
                      save_search_output_path='search_output.csv')

    return df_s


def _download_xml_file(xml_link):
    save_at = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files/web_xml/'

    saved_links = []
    t =len(xml_link)

    for i, link in enumerate(xml_link):
        file_name = link.split('/')[-1]
        s = save_at + file_name

        if os.path.isfile(s):
            saved_links.append(s)
            continue
        else:
            print(f"{i} /{t}  Downloading file:{file_name}")
            response = requests.get(link)
            with open(s, 'wb') as file:
                file.write(response.content)
            saved_links.append(s)
            #print(">>> %s downloaded!\n" % file_name)
    return saved_links


def _get_file(search_keywords, testing, extract_extire_database):
    base_dir = '/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files'
    if testing:
         #/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/xml_parser/xml_files/BaJoSe_BWV8_COM_6-6_ChoraleHer_TobisNo_00097.xml
        # path = base_dir + "/test_case_xml_parser_example.xml"
        # path = base_dir + "/test_case_tied_note.xml"
        # path = base_dir + "/ultimate_tie_test.xml"
        # path = base_dir + "/ultimate_tie_test2.xml"
        # path = base_dir + "/sacred_xml.xml"
        # path = base_dir + "/test_case_ives1.xml"
        # path = base_dir + "/ultimate_tie_test3.xml"
        # path = base_dir + "/ultimate_tie_test4.xml"
        # path = base_dir + "/ultimate_tie_test5.xml"
        # path = base_dir + "/voice_tie1.xml"
        # path = base_dir + "/BaJoSe_BWV8_COM_6-6_ChoraleHer_TobisNo_00097.xml"
        # path = base_dir + "/PrJode_Jos1102_COM_1-5_MissaLasol_002_00137.xml"
        # path = base_dir + "/breakme.xml"
        # path = base_dir + "/Untitled6.xml"
        # path = base_dir + "/stupid2.xml"
        # path = base_dir + "/BrumAn_Bru1011_COM_3-6_MissaProde_002_01134.xml"
        # path = base_dir + "/test_case_Castuski_1_time_sig_measure.xml"
        path = base_dir + "/MoWo_K80_COM_1-4_StringQuar_003_00838.xml"
        path = [path]

    else:
        df_s = _scrape_database(search_keywords, extract_extire_database)
        print(f"database shape {np.shape(df_s)}")
        urls = df_s['url'].to_list()
        path = _download_xml_file(urls)

    #assert os.path.isfile(path), "File not found {}".format(path)
    print("###### FIles", len(path))
    print("----------------PARSING--------------------")
    print(path)
    try:
        b = converter.parse(path)
        #b.show('text')
    except:
        print("ERROR parsing music21")
    return path

def plotting_wrapper(df):
    from hfm.scripts_in_progress.xml_parser.scripts.utils import _create_pianoroll_single, _create_pianoroll_single_parts
    offset = list(np.squeeze(df['Offset'].to_numpy(dtype=float)))
    duration = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
    midi = list(np.squeeze(df['MIDI'].to_numpy(dtype=int)))
    _create_pianoroll_single(pitch=midi, time=offset, duration =duration, midi_min=55, midi_max=75)


def plotting_wrapper_parts(df):

    from hfm.scripts_in_progress.xml_parser.scripts.utils import _create_pianoroll_single, _create_pianoroll_single_parts

    offset = list(np.squeeze(df['Offset'].to_numpy(dtype=float)))
    duration = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
    midi = list(np.squeeze(df['MIDI'].to_numpy(dtype=int)))
    measure = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
    partid = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))
    _create_pianoroll_single_parts(pitch=midi, time=offset, measure=measure, partid=partid,duration =duration, midi_min=55, midi_max=75)



if __name__ == "__main__":
    """
    Stabel Extractor! TO CHECEK duration 

    """

    correct_list = []
    error_list = []


    search_keywords = {'Composer': ['josquin'],
                       'Movement Number': None,
                       'Title': ['missa', ],
                       'Key': None,
                       'Life Time Year': None,
                       'Life Time Range': None,
                       'Year Range': None}

    paths = _get_file(search_keywords, testing=True, extract_extire_database=True)
    c= 0
    e=0
    for i, path in enumerate(paths):
        try:
            xml_tools = XMLToolBox(file_path=path)
            df_data = xml_tools.strip_xml()
            #print(df_data)
            if True:
                df_data_m = xml_tools.compute_measure_offset(df_data)
                df_data_v = xml_tools.compute_voice_offset(df_data_m)
                df_data_chord_tied = xml_tools.compute_tie_duration(df_data_v)
                df_data_midi = xml_tools.convert_pitch_midi(df_data_chord_tied)
                c+=1#

                correct_list.append(str(path))
                print(df_data_midi)

                #plotting_wrapper(df_data_midi)
                plotting_wrapper_parts(df_data_midi)
        except:
            error_list.append(str(path))
            e+=1
        print(f"{i}  Error {e} corr {c}, {os.path.basename(path)}")

    #print("TOTAL Files ERROR", len(error_list))



"""

001
002
003 - - S böck 
inharmonic information? 
"""