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
        self.measure_info = self.get_measure_info()



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
    def get_measure_info(self):
        for p in self.root.iter('measure'):
            print(f"measure {p.tag} {p.attrib['number']} ")

    def get_tie_information(self):
        c = 0
        step = []
        octave = []
        tie = []
        duration = []
        voice_num = []
        pitch_notes_v = [[] for i in range(self.num_voices)]
        octave_v = [[] for i in range(self.num_voices)]

        for p in self.root.iter('note'):
            c += 1
            tie_info = p.find('tie')

            if tie_info != None:
                tie.append(tie_info.attrib['type'])
            else:
                tie.append("none")
            # print(f"--- {p.find('voice').text}")


            for pp in p:
                if pp.tag == 'voice':
                    #print("VOICE", pp.text)
                    voice_num.append(pp.text)

                if pp.tag == 'duration':
                    duration.append(pp.text)

                if pp.tag == 'pitch':
                    for ppp in pp:
                        if ppp.tag == 'step':
                            step.append(ppp.text)

                        if ppp.tag == 'octave':
                            octave.append(ppp.text)
                if pp.tag == 'rest':
                    step.append(pp.tag)
                    octave.append(pp.tag)

        assert len(step) == len(octave)
        assert len(step) == len(tie)
        assert len(duration) == len(tie)

        notes = []

        for i in range(len(step)):
            n = str(step[i]) + str(octave[i])
            if n == "restrest":
                n = "rest"
            notes.append(n)
        df_tie_info = pd.DataFrame(np.array([duration, step, octave, tie]).T,
                                   columns=["Duration", 'Pitch', 'Octave', 'Tie Type'])
        return df_tie_info


class XMLHandler(xml.sax.ContentHandler):
    def __init__(self, file_path):
        self.file_path = self.pre_process_file(file_path)
        self.num_voices = self._check_voices()
        self.note_counter = 0
        self.time_axis = 0
        self.chord_available = False
        self.isdurationset = False
        self.isFitstNote = False
        self.curr_duration = 0
        self.curr_measure_duration = 0
        self.measure_offset = [0.0, ]
        self.curr_measure_num = 0
        self.note_info = u""
        self.curr_oct = u""

        self.pitch_list = []
        self.duration_list = []
        self.octave_list = []
        # remember to do ND array for n voices
        self.measure_num_list = []
        self.chord_info_list = []

    def _check_voices(self):

        pass

    def pre_process_file(self, file_path):
        """Simply consists in removing the DOCTYPE that make the xml parser crash
        """
        temp_file = tempfile.NamedTemporaryFile('w', suffix='.xml', prefix='tmp', delete=False, encoding='utf-8')

        # Remove the doctype line
        with open(file_path, 'r', encoding="utf-8") as fread:
            for line in fread:
                if not re.search(r'<!DOCTYPE', line):
                    temp_file.write(line)
        temp_file_path = temp_file.name
        temp_file.close()
        return temp_file_path

    def startElement(self, tag, attrib):
        self.tag = tag
        if self.tag == 'note':
            self.note_counter += 1
            print(f"............Start Note {self.note_counter}............................ ")

        if self.tag == 'part':
            self.part_id = attrib['id']
            print("-------PART {}-------".format(attrib['id']))

        if self.tag == "measure":
            self.m_num = attrib['number']
            print("############", self.tag, attrib['number'],
                  "############")
            self.curr_measure_num = int(attrib['number'])
        if self.tag == "tie":
            self.tie_status = attrib['type']

        if self.tag == "chord":
            self.chord_info_list.append("Chord")
            self.chord_available = True

        if self.tag == "sound":
            self.sound_tempo = attrib['tempo']
            print(f".. sound tempo :{self.sound_tempo}")

    def characters(self, content):
        if self.tag == 'step':
            self.note_name = content
        if self.tag == 'octave':
            self.octave = content
        if self.tag == 'duration':
            self.duration = content
        if self.tag == 'voice':
            self.voice = content
        if self.tag == 'tie':
            self.tie = content
        if self.tag == 'beats':
            self.beats = content
        if self.tag == 'beat-type':
            self.beat_type = content
        if self.tag == "divisions":
            self.divisions = int(content)
        if self.tag == "rest":
            self.rest_tag_stupid = content

    def endElement(self, textcontent):
        if self.tag == 'duration':
            self.curr_duration = float(float(self.duration) / float(self.divisions))
            self.isdurationset = True
            if self.note_counter == 1:
                pass
            else:
                self.time_axis += self.curr_duration

            print(f".. ## TIME axis {self.time_axis}")
            print("..", self.tag, "division :", self.divisions, "duration :", self.duration, "Actual Duration:",
                  self.curr_duration)
            self.duration_list.append(self.curr_duration)
        if self.tag == 'note':
            print(f"Start Note End {self.note_counter}")

        if self.tag == 'rest':
            print(f"..{self.tag} {self.measure_offset}")
            print(f".. curr_measure_num {self.curr_measure_num}")

            self.note_info = self.tag
            self.octave_list.append("")

        if self.tag == 'step':
            print("..", self.tag, ":", self.note_name)
            self.note_info = str(self.note_name)
            print(
                f".. {self.tag} {self.note_name} {self.curr_measure_duration} {self.curr_measure_num} {self.measure_offset}")
        if self.tag == 'octave':
            self.curr_oct = int(self.octave)
            self.octave_list.append(self.curr_oct)
            print(f".. {self.tag} : {self.octave} {self.curr_oct}")

        if self.tag == 'voice':
            print("..", self.tag, ":", self.voice)
        if self.tag == 'tie':
            print("..", self.tag, ":", self.tie_status)
        if self.tag == 'beats':
            print("..", self.tag, ":", self.beats)
        if self.tag == 'beat-type':
            print(f".. {self.tag} {self.beat_type}")
            print(f".. Measure Duration {float(4 * (float(self.beats) / float(self.beat_type)))}")
            self.curr_measure_duration += float(4 * (float(self.beats) / float(self.beat_type)))

            self.measure_offset.append(self.curr_measure_duration)

        if self.tag == "divisions":
            print("..", self.tag, ":", self.divisions)

        if self.tag == 'duration':
            print(f".. DURATION {self.duration}")

        if self.note_info != '':
            if self.chord_available == False:
                self.chord_info_list.append("xx")
            else:
                self.chord_available = False
            self.pitch_list.append(self.note_info)
            self.measure_num_list.append(int(self.curr_measure_num))

        self.note_info = ''
        self.curr_oct = ''
        self.tag = ''


def _get_file():
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_xml_parser_example.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_tied_note.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test2.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test3.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/sacred_xml.xml"
    # path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/test_case_ives1.xml"
    path = "/hfm/scripts_in_progress/example_files/stupid2.xml"
    b = converter.parse(path)
    b.show('text')
    return path


def _merge_measure_duration_and_get_offset(measure_offset, measure_num_list, duration_list, chord_info_list):
    offset_list = np.zeros(len(measure_num_list))
    offset_list = list(offset_list)
    num_m = np.max(measure_num_list)
    try:
        measure_offset.pop(num_m)
    except:
        pass
    # print(f"new measure {measure_offset} {len(measure_offset)}")

    idx_offsets = [measure_num_list.index(i + 1) for i in range(num_m)]
    # print(len(idx_offsets))
    assert len(idx_offsets) == len(measure_offset)

    for idx, i_off in enumerate(idx_offsets):
        offset_list[i_off] = measure_offset[idx]
    assert len(offset_list) == len(measure_num_list)
    """
        n_offset_list = []
        c_off = 0
        for i in range(len(duration_list)):
            if i==0:
                c_off = 0
            else:
                c_off += duration_list[i]
            n_offset_list.append(c_off)
            #print(f"{offset_list[i]} \t {c_off} \t {duration_list[i]}")
    """
    nn_offset_list = []
    c_off = 0

    for i, c in enumerate(chord_info_list):
        if i == 0:
            c_off = 0

        else:
            if c == 'Chord':
                temp_ch = []
                c_off = nn_offset_list[i - 1]
                if chord_info_list[i + 1] == 'xx':
                    pass
            else:
                c_off = nn_offset_list[i - 1] + duration_list[i - 1]

        nn_offset_list.append(c_off)

    return nn_offset_list


def _deal_with_ties():
    pass


if __name__ == "__main__":


    """
    

    score_path = "/Users/chris/DocumentLocal/workspace/hfm/scripts_in_progress/example_files/ultimate_tie_test3.xml"
    quantization = 8
    pianoroll, articulation = scoreToPianoroll(score_path, quantization)
    print(pianoroll)
    """
    path = _get_file()

    xml_tools = XMLToolBox(file_path=path)
    tie_info = xml_tools.get_tie_information()

"""

    hander = XMLHandler(file_path=path)
    parser = xml.sax.make_parser()

    parser.setContentHandler(hander)
    parser.parse(path)

    measure_offset = hander.measure_offset

    df_xml = []
    pitch_list = hander.pitch_list
    duration_list = hander.duration_list
    octave_list = hander.octave_list
    measure_num_list = hander.measure_num_list
    chord_info_list = hander.chord_info_list

    pitch_comb_list = []
    for id, m in enumerate(pitch_list):
        pitch_comb_list.append(str(m) + str(octave_list[id]))

    print("####################################################")
    print(
        f"measure_offset {len(measure_offset)} pitch {len(pitch_list)} duration {len(duration_list)} Oct {len(octave_list)}")
    print(
        f"measure num {len(measure_num_list)} pitch_comb_list {len(pitch_comb_list)} chord_info_list {len(chord_info_list)}")
    print("####################################################")

    # for id, m in enumerate(measure_offset):
    #    print(f"measure {id + 1} : {m}")

    # for id, m in enumerate(pitch_list):
    #   print(f"ck {id + 1} \t {duration_list[id]} \t  {str(m) + str(octave_list[id])} \t {measure_num_list[id]}")

    offset_list = _merge_measure_duration_and_get_offset(measure_offset=measure_offset,
                                                         measure_num_list=measure_num_list, duration_list=duration_list,
                                                         chord_info_list=chord_info_list)

    pd_data = pd.DataFrame(np.array(
        [offset_list, duration_list, pitch_comb_list, measure_num_list, chord_info_list, tie_info['Tie Type']]).T,
                           columns=['Offset', 'Duration', 'Pitch', 'Measure', 'chord_info', 'Tie Type'])

    print(pd_data)
    print(f"NUM VOICES : {num_voices}")


"""