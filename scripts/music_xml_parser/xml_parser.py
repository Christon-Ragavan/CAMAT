import re
import tempfile
import traceback
import xml.etree.ElementTree as ET
from os.path import isfile
import numpy as np
import pandas as pd
from utils import set_up_logger
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 1000000)
pd.set_option('display.width', 1000000)


def xml_parse(path,*args,**kwargs):
    logger = kwargs['logger']

    if isfile(path) == False:
        logger.info(f"File Not Found - {path}")
        raise FileExistsError ("File Does Note Exist".format(path))
    else:
        logger.info(f"File Found - {path}")
    xml_tools = XMLToolBox(file_path=path, logger=logger)
    df_data = xml_tools.strip_xml()
    df_data_m = xml_tools.compute_measure_offset(df_data)
    df_data_v = xml_tools.compute_voice_offset(df_data_m)
    df_data_chord_tied = xml_tools.compute_tie_duration(df_data_v)
    df_data_midi = xml_tools.convert_pitch_midi(df_data_chord_tied)
    df_data_f = xml_tools.remove_df_cols(df_data_midi)
    return df_data_f


class XMLToolBox:
    def __init__(self, file_path, *args,**kwargs):
        self.logger = kwargs['logger']

        self.file_path = self.pre_process_file(file_path)
        try:
            score = ET.parse(self.file_path)
            self.root = score.getroot()
        except Exception as error:
            traceback.print_exc()
            self.logger.error(error, exc_info=True)
            raise Exception("Unable to read .xml file")
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
        self.gracenote_tags = []

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

    def _find_chords(self, itt):
        chord_i = itt.find('chord')
        if chord_i != None:
            self.set_chord = True
            self.chord_tags.append('chord')
        else:
            self.set_chord = False
            self.chord_tags.append("none")

    def _find_voice(self, itt):
        voice_i = itt.find('voice')
        if voice_i != None:
            self.voice_num.append(voice_i.text)
        else:
            self.voice_num.append("0")

    def _find_note(self, itt):
        note_i = itt.find('note')
        if note_i != None:
            for p in note_i:
                try:
                    if p.tag == 'pitch':
                        note_tag = False
                    else:
                        note_tag = True
                except:
                    continue
        else:
            note_tag = False
        return note_tag

    def _find_duration(self, itt):
        duration_i = itt.find('duration')

        if duration_i == None:
            grace_i = itt.find('grace')
            if grace_i:
                if grace_i.attrib['slash'] == 'yes':
                    cd = 0.0
                    self.gracenote_tags.append('grace')
                else:
                    self.gracenote_tags.append('none')
                    cd = 0.0
            else:
                cd = 0.0
                self.gracenote_tags.append('none')

        else:
            self.gracenote_tags.append('none')
            cd = float(float(duration_i.text) / self.curr_measure_divisions)
        self.duration.append(cd)

    def _set_curr_measure_duration(self, itt):
        time_i = itt.find('time')
        if time_i != None:
            b = [None, None]
            for t in time_i:
                if t.tag == 'beats':
                    b[0] = float(t.text)
                elif t.tag == 'beat-type':
                    b[1] = float(t.text)
                elif t.tag == 'senza-misura':
                    raise Exception("senza-misura not implemented -> tag found:{}".format(t.tag))
                else:
                    raise Exception("Not implemented -> tag found:{}".format(t.tag))
            d = float(b[0] / b[1])
            self.curr_measure_offset = float(4 * d)
            self.measure_duration_dict.update(
                {str(self.curr_measure_num) + '_' + str(self.curr_part_id): self.curr_measure_offset})

    def _strip_part_information(self):
        pass

    def _get_curr_measure(self):
        pass

    def strip_xml(self):
        c = 0
        self.part_id_counter = 0
        self.measure_id_counter = 0
        for part in self.root.iter('part'):
            self.part_id_counter += 1
            self.measure_id_counter = 0
            self.logger.debug(f" ------------- part {part.tag} {part.attrib['id']} {self.part_id_counter}-------------")
            self.part_id_list.append(part.attrib['id'])
            self.curr_part_id = self.part_id_counter
            # self.curr_part_id = part.attrib['id']

            for m in part:
                self.measure_id_counter += 1
                self.logger.debug(f"measure {self.measure_id_counter}")
                self.curr_measure_num = self.measure_id_counter
                self.measure_number_list.append(self.curr_measure_num)
                if self.curr_measure_num == 1:
                    self.measure_offset_list.append(0.0)
                else:
                    self.measure_offset_list.append(self.curr_measure_offset)
                note_i = m.find('note')
                if note_i != None:
                    note_tag = True
                else:
                    note_tag = False
                if note_tag:
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
                            # self.logger.debug(f" # Note num {self.note_counter} -- {self.curr_measure_offset}")
                            self._find_ties(itt=m_itt)
                            self._find_chords(itt=m_itt)
                            self._find_voice(itt=m_itt)
                            self._find_duration(itt=m_itt)

                            for p in m_itt:  # itterate over pitches
                                c += 1
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
                else:
                    for m_itt in m:  # itterate over measure
                        if m_itt.tag == 'attributes':
                            division_i = m_itt.find('divisions')
                            self._set_curr_measure_duration(itt=m_itt)
                            if division_i != None:
                                self.curr_measure_divisions = float(division_i.text)

                    self.note_counter += 1
                    self.glb_part_id_list.append(self.curr_part_id)
                    self.note_counter_list.append(self.note_counter)
                    self.measure_num_per_note_list.append(self.curr_measure_num)
                    self._find_ties(itt=m_itt) # TODO: TESTTHIS
                    self._find_chords(itt=m_itt)
                    self._find_voice(itt=m_itt)
                    self.duration.append(self.curr_measure_offset)
                    self.gracenote_tags.append('none')
                    self.step.append('rest')
                    self.octave.append('rest')

        if True:
            self.logger.debug(f"self.step                   :{len(self.step)}")
            self.logger.debug(f"self.octave                 :{len(self.octave)}")
            self.logger.debug(f"self.tie                    :{len(self.tie)}")
            self.logger.debug(f"self.duration               :{len(self.duration)}")
            self.logger.debug(f"self.chord_tags             :{len(self.chord_tags)}")
            self.logger.debug(f"self.voice_num              :{len(self.voice_num)}")
            self.logger.debug(f"curr_measure_num            :{self.curr_measure_num}")
            self.logger.debug(f"measure_offset_list         :{len(self.measure_offset_list)}")
            self.logger.debug(f"self.GraceNote              :{len(self.gracenote_tags)}")
            self.logger.debug(f"self.note_counter_list      :{len(self.note_counter_list)}")
            self.logger.debug(f"self.duration               :{len(self.duration)}")
            self.logger.debug(f"self.step                   :{len(self.step)}")
            self.logger.debug(f"self.octave                 :{len(self.octave)}")
            self.logger.debug(f"self.measure_num_per_note   :{len(self.measure_num_per_note_list)}")
            self.logger.debug(f"self.voice_num              :{len(self.voice_num)}")
            self.logger.debug(f"self.glb_part_id_list       :{len(self.glb_part_id_list)}")
            self.logger.debug(f"self.chord_tags             :{len(self.chord_tags)}")
            self.logger.debug(f"self.tie                    :{len(self.tie)}")

        assert len(self.step) == len(self.octave) == len(self.tie) == len(self.duration) == len(self.gracenote_tags)
        assert len(self.step) == len(self.tie)
        assert len(self.duration) == len(self.tie)
        assert len(self.duration) == len(self.chord_tags)
        assert len(self.duration) == len(self.glb_part_id_list)

        if len(self.voice_num) == 0:
            self.voice_num = [1] * len(self.duration)

        try:
            df_data = pd.DataFrame(np.array(
                [self.note_counter_list, self.duration, self.step, self.octave,
                 self.measure_num_per_note_list, self.voice_num, self.glb_part_id_list,
                 self.chord_tags,
                 self.tie, self.gracenote_tags]).T,
                                   columns=["#Note_Debug", "Duration", 'Pitch', 'Octave', 'Measure', 'Voice', 'PartID',
                                            'Chord Tags',
                                            'Tie Type', 'Grace Tag'])
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
                self.logger.debug("error in _pitch_to_midi()", pitch)
                self.logger.debug(f"error in _pitch_to_midi {pitch}")

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

        t_pl = [self._pitch_to_midi(octave_list[i], t_pl[i]) for i in range(len(t_pl))]
        self.midi_list = [t_pl[i] + t_shifter[i] if t_pl[i] != None else None for i in range(len(t_pl))]

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
                        l_df_start.iloc[0, l_df_start.columns.get_loc('Duration')] = l_duration
                        self._tie_append_wrapper(df, ii, l_df_start)
                        df.iloc[ii, df.columns.get_loc('Tie Type')] = 'SKIPPED_stop'
                        break
                pass
        return self.df_data_tie

    def _measure_offset_sparse(self, measure_num_list, partid_num_list, measure_offset, idx_new_measure_offsets):
        if len(idx_new_measure_offsets) != len(measure_offset):
            self.logger.debug(f"Shape not equal idx_new_measure_offsets: {len(idx_new_measure_offsets)} and measure_offset: {measure_offset}")

        assert len(idx_new_measure_offsets) == len(measure_offset)
        measure_offset_sum = []
        n_sparce_m = []
        n_measure_change_idx = []
        c = 0
        for i in range(1, len(measure_offset) + 1):
            if i < len(measure_offset):
                measure_offset_sum.append(np.sum(measure_offset[:i]))

                if idx_new_measure_offsets[i] == 1:
                    measure_offset[:i] = [0] * i
            else:
                measure_offset_sum.append(np.sum(measure_offset))

        assert len(measure_offset_sum) == len(
            measure_offset), f"Error in generating measure offset sum, measure_offset_sum:  {len(measure_offset_sum)}, measure_offset: {len(measure_offset)}"
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

    def _compute_idx_new_measure_for_multi_parts(self, measure_num_list):
        idx_new_measure_offsets = []
        for i, m in enumerate(measure_num_list):
            if i == 0:
                m_t = m
            else:
                if measure_num_list[i - 1] != m:
                    m_t = m
                else:
                    continue
            idx_new_measure_offsets.append(m_t)

        return idx_new_measure_offsets

    def compute_measure_offset(self, df):
        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        partid_num_list = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))

        idx_new_measure_offsets = self._compute_idx_new_measure_for_multi_parts(
            measure_num_list)  # computing number of measure needed in list

        assert len(idx_new_measure_offsets) == len(
            self.measure_offset_list), "Check the lengths len(idx_new_measure_offsets){} !=len(measure_offset) {}".format(
            len(idx_new_measure_offsets), len(self.measure_offset_list))
        self.measure_off_sparse, self.n_measure_change_idx, measure_offset_sum = self._measure_offset_sparse(
            measure_num_list, partid_num_list,
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
        n_num_voice = max(max(list(set(voices_list))), 1)
        nn_offset_list = []
        voice_track_container = [[] for i in range(n_num_voice)]

        c_off = 0.0
        curr_measure_offset = 0.0
        measure_C = 0

        for i, v in enumerate(voices_list):
            c_ch = chord_info_list[i]
            m_trk = measure_num_list[i]
            pid_trk = part_id_list[i]
            c_dur = duration_list[i]
            if i in self.n_measure_change_idx:  # resetting the container at every change in measure
                measure_C += 1
                curr_measure_offset = self.measure_off_sparse[i]
                del voice_track_container
                voice_track_container = [[] for i in range(n_num_voice)]

            if part_id_list[i - 1] != pid_trk:  # resetting the container at every change in measure
                c_off = 0.0
            if i == 0:
                c_off = 0.0
            else:
                if part_id_list[i - 1] != pid_trk:  # resetting the container at every change in measure
                    c_off = 0.0
                    del voice_track_container
                    voice_track_container = [[] for i in range(n_num_voice)]
                elif voices_list[i - 1] != v:  # if there is a change in voice do this
                    if measure_num_list[i - 1] != m_trk:
                        c_off = curr_measure_offset
                    elif len(voice_track_container[v - 1]) == 0:
                        c_off = curr_measure_offset
                    else:
                        c_off = np.sum(voice_track_container[v - 1][-1])
                else:
                    if measure_num_list[i - 1] != m_trk:
                        c_off = curr_measure_offset
                    elif c_ch == 'chord':
                        c_off = nn_offset_list[i - 1]
                    else:
                        c_off = nn_offset_list[i - 1] + duration_list[i - 1]
            voice_track_container[v - 1].append([c_off, c_dur])
            nn_offset_list.append(c_off)
        df.insert(loc=2, column='Offset', value=nn_offset_list)
        return df

    def compute_chord_offset(self, df):
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['Chord Tags'].to_numpy()))
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

    def remove_df_cols(self, df, drop_colms_labels=None):
        if drop_colms_labels is None:
            drop_colms_labels = ['#Note_Debug', 'Offset_ml']
        return df.drop(drop_colms_labels, axis=1)
