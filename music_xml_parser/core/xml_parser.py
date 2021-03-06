import copy
import re
import tempfile
import traceback
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from os.path import isdir, basename, isfile,join

np.seterr(all="ignore")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    from .parser_utils import set_up_logger, _inseart_row_in_pd
except:
    from parser_utils import set_up_logger, _inseart_row_in_pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

class XMLToolBox:
    def __init__(self, file_path, *args, **kwargs):
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

        self.curr_time_signature = ''
        self.curr_time_signature_adj = ''
        self.time_signature_list = []
        self.time_signature_list_adj = []
        self.time_onset = []
        self.part_id_list = []
        self.part_name_list = []
        self.glb_part_id_list = []
        self.curr_part_id = None
        self.note_counter = 0
        self.note_counter_list = []
        self.df_data_tie = pd.DataFrame()
        self.measure_duration_dict = dict()
        self.measure_duration_list = []
        self.measure_onset_list = []

        self.step = []
        self.octave = []
        self.tie = []
        self.duration = []
        self.voice_num = []
        self.chord_tags = []
        self.gracenote_tags = []
        self.measure_onset_df = []
        self.measure_number_list_corr_part = []

        self.set_chord = False
        self.measure_number_list = []
        self.measure_num_per_note_list = []
        self.curr_measure_num = 0
        self.voices_nd_list = [[] for i in range(self.num_voices)]
        self.voices = []
        self.curr_measure_onset = 0.0
        self.curr_measure_onset_df = 0.0

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
        """
        4 * (4/4) = 4
        4 * (2/4) = 2
        4 * (2/1) = 8
        """
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

            self.curr_time_signature = str(int(float(b[0]))) + '/' + str(int(float(b[1])))
            self.curr_time_signature_adj = str(4 * int(float(b[0]))) + '/' + str(4 * int(float(b[1])))
            self.curr_measure_onset = float(4 * d)

    def _strip_part_information(self):
        pass

    def _get_curr_measure(self):
        pass

    def _get_curr_part_name(self, part_id):
        pl = self.root.iter('part-list')
        part_name = 'none'
        for pl_d in pl:
            for i in pl_d:
                if i.tag == 'score-part':
                    if part_id == i.attrib['id']:
                        pn = i.find('part-name')
                        part_name = str(pn.text)
        return part_name

    def strip_xml(self):
        c = 0
        self.part_id_counter = 0
        self.measure_id_counter = 0
        self.curr_measure_onset_df = 0.0
        self.curr_measure_duration_list = 0.0
        for part in self.root.iter('part'):
            self.part_id_counter += 1
            self.measure_id_counter = 0
            self.logger.debug(f" ------------- part {part.tag} {part.attrib['id']} {self.part_id_counter}-------------")
            self.part_id_list.append(part.attrib['id'])
            self.curr_part_id = self.part_id_counter
            curr_part_name = self._get_curr_part_name(part.attrib['id'])
            # self.curr_part_id = part.attrib['id']

            for idx_t, m in enumerate(part):
                self.logger.debug(
                    f"measure {self.measure_id_counter} measure idtag {int(m.attrib['number'])}  curr_onset {self.curr_measure_onset}")
                if idx_t == 0 and int(m.attrib['number']) == 0:
                    pass
                else:
                    self.measure_id_counter += 1
                self.curr_measure_num = copy.deepcopy(self.measure_id_counter)
                # self.curr_measure_num = int(m.attrib['number'])
                self.measure_number_list.append(self.curr_measure_num)
                self.measure_number_list_corr_part.append(self.curr_part_id)

                if idx_t == 0:
                    # if self.measure_number_list[0] == 1 or self.measure_number_list[0] == 0:
                    self.measure_onset_list.append(0.0)
                    self.curr_measure_onset_df = 0.0
                else:
                    self.curr_measure_onset_df += self.curr_measure_onset
                    self.measure_onset_list.append(self.curr_measure_onset)

                note_i = m.find('note')

                if note_i != None:
                    note_tag = True
                else:
                    note_tag = False

                if note_tag:
                    for m_itt in m:  # itterate over measure
                        self.measure_duration_dict.update(
                            {str(self.curr_measure_num) + '_' + str(self.curr_part_id): self.curr_measure_onset})

                        if m_itt.tag == 'attributes':
                            division_i = m_itt.find('divisions')
                            self._set_curr_measure_duration(itt=m_itt)
                            if division_i != None:
                                self.curr_measure_divisions = float(division_i.text)

                        if m_itt.tag == 'note':
                            self.note_counter += 1
                            self.measure_duration_list.append(self.curr_measure_onset)
                            self.time_signature_list.append(self.curr_time_signature)
                            self.time_signature_list_adj.append(self.curr_time_signature_adj)
                            self.glb_part_id_list.append(self.curr_part_id)
                            self.part_name_list.append(curr_part_name)
                            self.measure_onset_df.append(self.curr_measure_onset_df)

                            self.note_counter_list.append(self.note_counter)
                            self.measure_num_per_note_list.append(self.curr_measure_num)
                            # self.logger.debug(f" # Note num {self.note_counter} -- {self.curr_measure_onset}")
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
                        self.measure_duration_dict.update(
                            {str(self.curr_measure_num) + '_' + str(self.curr_part_id): self.curr_measure_onset})

                        if m_itt.tag == 'attributes':
                            division_i = m_itt.find('divisions')
                            self._set_curr_measure_duration(itt=m_itt)
                            if division_i != None:
                                self.curr_measure_divisions = float(division_i.text)

                    self.note_counter += 1
                    self.measure_duration_list.append(self.curr_measure_onset)
                    self.time_signature_list.append(self.curr_time_signature)
                    self.time_signature_list_adj.append(self.curr_time_signature_adj)
                    self.measure_onset_df.append(self.curr_measure_onset_df)
                    self.glb_part_id_list.append(self.curr_part_id)
                    self.part_name_list.append(curr_part_name)
                    self.note_counter_list.append(self.note_counter)
                    self.measure_num_per_note_list.append(self.curr_measure_num)
                    self._find_ties(itt=m_itt)  # TODO: TESTTHIS
                    self._find_chords(itt=m_itt)
                    self._find_voice(itt=m_itt)
                    self.duration.append(self.curr_measure_onset)
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
            self.logger.debug(f"self.GraceNote              :{len(self.gracenote_tags)}")
            self.logger.debug(f"self.measure_num_per_note   :{len(self.measure_num_per_note_list)}")
            self.logger.debug(f"self.glb_part_id_list       :{len(self.glb_part_id_list)}")
            self.logger.debug(f"self.part_name_list         :{len(self.part_name_list)}")
            self.logger.debug(f"measure_onset_list          :{len(self.measure_onset_list)}")
            self.logger.debug(f"self.note_counter_list      :{len(self.note_counter_list)}")
            self.logger.debug(f"self.measure_onset_df       :{len(self.measure_onset_df)}")
            self.logger.debug(f"self.time_signature_list    :{len(self.time_signature_list)}")
            self.logger.debug(f"measure_duration_list       :{len(self.measure_duration_list)}")
            self.logger.debug(f"curr_measure_num            :{self.curr_measure_num}")

        assert len(self.step) == len(self.octave) == len(self.tie) == len(self.duration) == len(self.gracenote_tags), \
            f"\nstep             :{len(self.step)}\noctave           :{len(self.octave)}\ntie              :{len(self.tie)}\nduration         :{len(self.duration)}\ngracenote_tags   :{len(self.gracenote_tags)} "

        assert len(self.duration) == len(self.tie)
        assert len(self.duration) == len(self.measure_onset_df)
        assert len(self.duration) == len(self.chord_tags)
        assert len(self.duration) == len(self.glb_part_id_list)

        if len(self.voice_num) == 0:
            self.voice_num = [1] * len(self.duration)

        df_data = pd.DataFrame(np.array(
            [self.note_counter_list,
             self.duration,
             self.step,
             self.octave,
             self.measure_num_per_note_list,
             self.voice_num,
             self.glb_part_id_list,
             self.part_name_list,
             self.chord_tags,
             self.tie,
             self.gracenote_tags,
             self.measure_onset_df,
             self.measure_duration_list,
             self.time_signature_list,
             self.time_signature_list_adj]).T,
                               columns=["#Note_Debug",
                                        "Duration",
                                        'Pitch',
                                        'Octave',
                                        'Measure',
                                        'Voice',
                                        'PartID',
                                        'PartName',
                                        'ChordTag',
                                        'TieType',
                                        'GraceTag',
                                        'MeasureOnset',
                                        'MeasureDuration',
                                        'TimeSignature',
                                        'TimeSignatureAdjusted'])
        set_dtypes = {'#Note_Debug': int,
                      'Duration': float,
                      'Pitch': str,
                      'Octave': str,
                      'Measure': int,
                      'Voice': int,
                      'PartID': int,
                      'PartName': str,
                      'ChordTag': str,
                      'TieType': str,
                      'GraceTag': str,
                      'MeasureOnset': float,
                      'MeasureDuration': float,
                      'TimeSignature': str,
                      'TimeSignatureAdjusted': str
                      }
        df_data = df_data.astype(set_dtypes)
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
        tie_info = list(set(list(df['TieType'].to_numpy())))
        if 'start' not in tie_info:
            return df
        else:

            for i in range(t_len):
                curr_tie_typ = df['TieType'][i]
                # TODO: write some error handeler
                set_dtypes = {'#Note_Debug': str,
                              'Duration': float,
                              'Pitch': str,
                              'Octave': str,
                              'Measure': int,
                              'Voice': str,
                              'PartID': str,
                              'PartName': str,
                              'ChordTag': str,
                              'TieType': str,
                              'GraceTag': str,
                              'MeasureOnset': float,
                              'TimeSignature': str,
                              'TimeSignatureAdjusted': str
                              }
                df = df.astype(set_dtypes)
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
                        ck_type = df['TieType'][ii]
                        try:
                            ck_curr_cont = df['Pitch'][ii] + df['Octave'][ii] + df['Voice'][ii]
                        except:
                            ck_curr_cont = df['Pitch'][ii] + df['Octave'][ii] + df['PartID'][ii]

                        if curr_cont == ck_curr_cont and ck_type == 'continue':
                            l_duration += float(df['Duration'][ii])
                            df.iloc[ii, df.columns.get_loc('TieType')] = 'SKIPPED_continue'

                        if curr_cont == ck_curr_cont and ck_type == 'stop':
                            l_duration += float(df['Duration'][ii])
                            l_df_start.iloc[0, l_df_start.columns.get_loc('Duration')] = l_duration
                            self._tie_append_wrapper(df, ii, l_df_start)
                            df.iloc[ii, df.columns.get_loc('TieType')] = 'SKIPPED_stop'
                            break
                    pass
            return self.df_data_tie

    def _measure_onset_sparse(self, measure_num_list, partid_num_list, measure_onset_g, idx_new_measure_onsets):
        measure_onset = measure_onset_g.copy()
        if len(idx_new_measure_onsets) != len(measure_onset):
            self.logger.debug(
                f"Shape not equal idx_new_measure_onsets: {len(idx_new_measure_onsets)} and measure_onset: {measure_onset}")

        assert len(idx_new_measure_onsets) == len(measure_onset)
        measure_onset_sum = []
        n_sparce_m = []
        n_measure_change_idx = []
        c = 0

        for i in range(1, len(measure_onset) + 1):
            if i < len(measure_onset):
                measure_onset_sum.append(np.sum(measure_onset[:i]))

                if idx_new_measure_onsets[i] == 1:
                    measure_onset[:i] = [0] * i
            else:
                measure_onset_sum.append(np.sum(measure_onset))
        assert len(measure_onset_sum) == len(
            measure_onset), f"Error in generating MeasureOnset sum, measure_onset_sum:  {len(measure_onset_sum)}, measure_onset: {len(measure_onset)}"

        for i, m in enumerate(measure_num_list):
            try:
                if m == idx_new_measure_onsets[c]:
                    n_measure_change_idx.append(i)
                    curr_measure = measure_onset_sum[c]
                    n_sparce_m.append(curr_measure)
                    c += 1
                else:
                    n_sparce_m.append(curr_measure)
            except:
                n_sparce_m.append(curr_measure)

                continue

        return n_sparce_m, n_measure_change_idx, measure_onset_sum

    def _compute_idx_new_measure_for_multi_parts(self, measure_num_list):
        idx_new_measure_onsets = []
        for i, m in enumerate(measure_num_list):
            if i == 0:
                m_t = m
            else:
                if measure_num_list[i - 1] != m:
                    m_t = m
                else:
                    continue
            idx_new_measure_onsets.append(m_t)

        return idx_new_measure_onsets

    def compute_measure_onset(self, df):

        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        partid_num_list = list(np.squeeze(df['PartID'].to_numpy(dtype=int)))

        idx_new_measure_onsets = self._compute_idx_new_measure_for_multi_parts(
            measure_num_list)  # computing number of measure needed in list


        assert len(idx_new_measure_onsets) == len(
            self.measure_onset_list), "Check the lengths len(idx_new_measure_onsets){} !=len(measure_onset) {}".format(
            len(idx_new_measure_onsets), len(self.measure_onset_list))

        self.measure_off_sparse, self.n_measure_change_idx, measure_onset_sum = self._measure_onset_sparse(
            measure_num_list,
            partid_num_list,
            self.measure_onset_list,
            idx_new_measure_onsets)

        df['MeasureOnset_ref'] = self.measure_off_sparse
        # df.insert(loc=1, column='MeasureOnset_ref', value=self.measure_off_sparse)
        return df

    def compute_voice_onset(self, df):
        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        MeasureOnset_ref_num_list = list(np.squeeze(df['MeasureOnset_ref'].to_numpy(dtype=int)))
        voices_list = list(np.squeeze(df['Voice'].to_numpy(dtype=int)))
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['ChordTag'].to_numpy()))
        part_id_list = list(np.squeeze(df['PartID'].to_numpy()))
        n_num_voice = max(max(list(set(voices_list))), 1)
        nn_onset_list = []
        voice_track_container = [[] for i in range(n_num_voice)]

        c_off = 0.0
        curr_measure_onset = 0.0
        for i, v in enumerate(voices_list):
            c_ch = chord_info_list[i]
            m_trk = measure_num_list[i]
            pid_trk = part_id_list[i]
            c_dur = duration_list[i]

            if i in self.n_measure_change_idx:  # resetting the container at every change in measure
                curr_measure_onset = self.measure_off_sparse[i]
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
                        c_off = curr_measure_onset
                    elif len(voice_track_container[v - 1]) == 0:
                        c_off = curr_measure_onset
                    else:
                        c_off = np.sum(voice_track_container[v - 1][-1])
                    if m_trk == 0:
                        c_off = 0.0
                else:
                    if measure_num_list[i - 1] != m_trk:
                        c_off = curr_measure_onset
                    elif c_ch == 'chord':
                        c_off = nn_onset_list[i - 1]
                    else:
                        c_off = nn_onset_list[i - 1] + duration_list[i - 1]
            voice_track_container[v - 1].append([c_off, c_dur])
            nn_onset_list.append(c_off)

        df['Onset'] = nn_onset_list
        return df

    def compute_chord_onset(self, df):
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['ChordTag'].to_numpy()))
        nn_onset_list = []
        for i, c in enumerate(chord_info_list):
            if i == 0:
                c_off = 0
            else:
                if c == 'chord':
                    c_off = nn_onset_list[i - 1]
                else:
                    c_off = nn_onset_list[i - 1] + duration_list[i - 1]
            nn_onset_list.append(c_off)
        df.insert(loc=3, column='onset_cl', value=nn_onset_list)
        return df
    def _shift_local_onset(self, df):
        local_o = df['LocalOnset'].tolist()
        local_o = list(map(lambda x:x+1, local_o))
        df['LocalOnset'] = local_o
        return df
    def remove_df_cols(self, df, drop_colms_labels=None):

        column_names = ["#Note_Debug",
                        "Onset",
                        "Duration",
                        "Pitch",
                        "Octave",
                        "MIDI",
                        "Measure",
                        "LocalOnset",
                        "Voice",
                        "PartID",
                        "PartName",
                        "MeasureOnset",
                        "MeasureOnset_ref",
                        "MeasureDuration",
                        "MeasureDurrDiff",
                        "TimeSignature",
                        "TimeSignatureAdjusted",
                        "Upbeat",
                        "ChordTag",
                        "TieType",
                        "GraceTag" ]
        df = df.reindex(columns=column_names)

        if drop_colms_labels is None:
            drop_colms_labels = ['#Note_Debug', 'MeasureOnset_ref'] ##Note_Debug
        df = df.drop(drop_colms_labels, axis=1)
        set_dtypes = {'Duration': float,
                      'Pitch': str,
                      'Octave': str,
                      'Measure': int,
                      'LocalOnset': float,
                      'Voice': int,
                      'PartID': int,
                      'PartName': str,
                      'ChordTag': str,
                      'TieType': str,
                      'GraceTag': str,
                      'MeasureOnset': float,
                      'MeasureDurrDiff': float,
                      'TimeSignature': str,
                      'TimeSignatureAdjusted': str,
                      'Upbeat':str
                      }

        df_data = df.astype(set_dtypes)
        # df_data = df_data['MeasureDurrDiff'].apply(lambda x: '%.3f' % x).values.tolist()
        return df_data


    def _compute_measure_n_onset(self):
        return pd.DataFrame(
            np.array([self.measure_number_list, self.measure_number_list_corr_part, self.measure_onset_list]).T,
            columns=['Measure Number', 'Part ID', 'MeasureOnset'])

    def _get_upbeat_measure_rest_info(self, df_c, upbeat_measure_dur):
        grouped = df_c.groupby(df_c.Measure)
        f_m = grouped.get_group(0).copy()
        moff2 = f_m.filter(['ChordTag', 'Duration', 'PartID', 'Voice'])
        convert_dict = {'Duration': float, 'PartID': int, 'Voice': int, 'ChordTag': str}
        moff2 = moff2.astype(convert_dict)
        moff2 = moff2[~moff2['ChordTag'].str.contains("chord")]
        up_beat_dur_per_part = moff2.groupby(["PartID", "Voice"]).agg({"Duration": np.sum}).reset_index()
        d = upbeat_measure_dur - up_beat_dur_per_part['Duration']
        up_beat_dur_per_part['rest_info'] = d

        return up_beat_dur_per_part

    def _re_compute_onset_upbeat(self, df):
        global curr_durr, t_on, curr_measure_onset

        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        voices_list = list(np.squeeze(df['Voice'].to_numpy(dtype=int)))
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['ChordTag'].to_numpy()))
        part_id_list = list(np.squeeze(df['PartID'].to_numpy()))

        n_num_voice = max(max(list(set(voices_list))), 1)
        nn_onset_list = []
        voice_track_container = [[] for i in range(n_num_voice)]
        MeasureOnset = list(np.squeeze(df['MeasureOnset'].to_numpy()))

        for i, v in enumerate(voices_list):
            c_ch = chord_info_list[i]
            m_trk = measure_num_list[i]

            pid_trk = part_id_list[i]
            c_dur = duration_list[i]

            if i in self.n_measure_change_idx:  # resetting the container at every change in measure
                curr_measure_onset = MeasureOnset[i]
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
                        c_off = curr_measure_onset
                    elif len(voice_track_container[v - 1]) == 0:
                        c_off = curr_measure_onset

                    else:
                        c_off = np.sum(voice_track_container[v - 1][-1])
                    if m_trk == 0:
                        c_off = 0.0
                else:
                    if measure_num_list[i - 1] != m_trk:
                        c_off = curr_measure_onset

                    elif c_ch == 'chord':
                        c_off = nn_onset_list[i - 1]
                    else:
                        c_off = nn_onset_list[i - 1] + duration_list[i - 1]
            voice_track_container[v - 1].append([c_off, c_dur])
            nn_onset_list.append(c_off)

        df['Onset'] = nn_onset_list
        return df

    def check_upbeat(self, df):
        up_beat_dur_per_measure = df.groupby(["Measure", "Voice", "PartID"]).agg({"Duration": np.sum}).reset_index()
        pass

    def testing_compute_voice_onset(self, df):
        measure_num_list = list(np.squeeze(df['Measure'].to_numpy(dtype=int)))
        voices_list = list(np.squeeze(df['Voice'].to_numpy(dtype=int)))
        duration_list = list(np.squeeze(df['Duration'].to_numpy(dtype=float)))
        chord_info_list = list(np.squeeze(df['ChordTag'].to_numpy()))
        part_id_list = list(np.squeeze(df['PartID'].to_numpy()))
        n_num_voice = max(max(list(set(voices_list))), 1)
        nn_onset_list = []
        voice_track_container = [[] for i in range(n_num_voice)]
        c_off = 0.0
        curr_measure_onset = 0.0
        for i, v in enumerate(voices_list):
            c_ch = chord_info_list[i]
            m_trk = measure_num_list[i]
            pid_trk = part_id_list[i]
            c_dur = duration_list[i]

            if i in self.n_measure_change_idx:  # resetting the container at every change in measure
                curr_measure_onset = self.measure_off_sparse[i]
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
                        c_off = curr_measure_onset
                    elif len(voice_track_container[v - 1]) == 0:
                        c_off = curr_measure_onset

                    else:
                        c_off = np.sum(voice_track_container[v - 1][-1])
                    if m_trk == 0:
                        c_off = 0.0
                else:
                    if measure_num_list[i - 1] != m_trk:
                        c_off = curr_measure_onset
                    elif c_ch == 'chord':
                        c_off = nn_onset_list[i - 1]
                    else:
                        c_off = nn_onset_list[i - 1] + duration_list[i - 1]
            voice_track_container[v - 1].append([c_off, c_dur])
            nn_onset_list.append(c_off)
        df.insert(loc=2, column='Onset', value=nn_onset_list)
        return df

    def check_for_upbeat_measure(self, df):
        df['UpbeatMeasure'] = ['none' for _ in range(len(df))]

        # Consider only one voice -> "1", in case of no voice "1", then search for "0"
        for i in [ "1", "0"]:
            gp = df.groupby(["Measure", "Voice", "PartID"]).agg({"Duration": np.sum}).reset_index()

            set_dtypes = {'Voice': str}
            gp = gp.astype(set_dtypes)
            gp = gp[gp['Voice'].str.contains(i)].reset_index()
            if len(gp)==0:
                continue
            else:
                break
        set_dtypes = {'Measure': int,
                      'Voice': int,
                      'PartID': int,
                      'Duration': float}
        gp = gp.astype(set_dtypes)
        # assert len(self.measure_duration_dict) == len(gp),f"Asserting the length of unique measure duration with pd " \
        # f"measure measure_duration_dict:{len(self.measure_duration_dict)}, GP:{len(gp)}"
        measure_durr_data = []
        for md in self.measure_duration_dict:
            c_mea, c_part = md.split('_', 1)
            n = np.squeeze(np.where(((gp['Measure'] == int(c_mea)) & (gp['PartID'] == int(c_part)))))
            if n.size == 1:
                gp.loc[int(n), 'MeasureDuration'] = 0
                gp.loc[int(n), 'MeasureDuration'] = self.measure_duration_dict[md]
                measure_durr_data.append([c_mea, c_part, self.measure_duration_dict[md]])
            else:
                continue
        gp['diff'] = gp['MeasureDuration'] - gp['Duration']
        gp['UpbeatTag'] = ['none' for _ in range(len(gp))]

        diff_tag = np.squeeze(np.where(( gp['diff'] != 0.0 )))
        if diff_tag.size !=0:
            for x in diff_tag:
                gp.loc[int(x), 'UpbeatTag']='True'

        self.upbeat_measure_info = gp.copy()
        column_names = ["#Note_Debug",
                        "Onset",
                        "Duration",
                        "Pitch",
                        "Octave",
                        "MIDI",
                        "Measure",
                        "Voice",
                        "PartID",
                        "PartName",
                        "MeasureOnset",
                        "MeasureOnset_ref",
                        "TimeSignature",
                        "TimeSignatureAdjusted",
                        "Upbeat",
                        "UpbeatMeasure",
                        "ChordTag",
                        "TieType",
                        "GraceTag"]
        df = df.reindex(columns=column_names)
        return df

    def _recompute_df_upbeat_mid(self, ub_st_idx, ub_en_idx,index_to_reset, df_c):
        cn = ['#Note_Debug',
              'Onset',
              'Duration',
              'Measure',
              'LocalOnset',
              'PartID',
              'Voice',
              'Pitch',
              'Octave',
              'MIDI',
              'MeasureOnset_ref',
              'Upbeat',
              'MeasureDurrDiff',
              'PartName',
              'ChordTag',
              'TieType',
              'GraceTag',
              'MeasureOnset',
              'MeasureDuration',
              'TimeSignature',
              'TimeSignatureAdjusted']

        df_c = df_c.reindex(columns=cn)

        assert ub_st_idx<index_to_reset<ub_en_idx, f"Resetting index not in range"
        reseting_idx_list = np.arange(index_to_reset-1, ub_en_idx)
        c = 0
        for i in reseting_idx_list:

            if i+1 <= ub_en_idx:
                # ub_mea_onset_diff = 0
                n_onset = df_c.loc[i, 'Onset'] + df_c.loc[i, 'Duration']
                df_c.loc[i + 1, 'Onset'] = n_onset
                df_c.loc[i + 1, 'Measure'] = int(df_c.loc[i + 1, 'Measure']) - 1
                if df_c.loc[i+1, 'Upbeat'] == True:
                    ub_mea_onset_diff = df_c.loc[i + 1, 'MeasureOnset'] - df_c.loc[i, 'MeasureOnset']
                    if df_c.loc[i+1, 'LocalOnset'] == 0.0:
                        if df_c.loc[i, 'Measure']  ==df_c.loc[i+1, 'Measure']:
                            df_c.loc[i+1, 'LocalOnset'] = df_c.loc[i, 'LocalOnset']+df_c.loc[i, 'Duration']



                df_c.loc[i+1, 'MeasureOnset'] = int(df_c.loc[i+1, 'MeasureOnset']) - ub_mea_onset_diff


        return df_c


    def compute_upbeat(self, df):
        df['Upbeat'] = ['none' for _ in range(len(df))]
        df = self.check_for_upbeat_measure(df)
        df_c = df.copy()
        meau = np.squeeze(df_c[['Measure']].to_numpy())
        if 0 in meau or '0' in meau:
            df_c.drop_duplicates(subset=['Measure',
                                         'PartID',
                                         'MeasureOnset'],
                                 keep="last", inplace=True)
            moff = np.squeeze(df_c[['MeasureOnset']].to_numpy(dtype=float))

            upbeat_measure_dur = moff[1] - moff[0]
            ub_rest_df = self._get_upbeat_measure_rest_info(df, upbeat_measure_dur)
            ub_r_np = ub_rest_df[["PartID", "Voice", "rest_info"]].to_numpy()
            convert_dict = {'Duration': float, 'PartID': int, 'Voice': int, 'ChordTag': str}
            df = df.astype(convert_dict)

            for i in range(np.shape(ub_r_np)[0]):
                m_idx = min(df[(df['PartID'] == ub_r_np[i][0]) & (df['Voice'] == ub_r_np[i][1])].index.tolist())
                n_v = df.iloc[[m_idx]].copy()
                n_v.loc[:, 'Pitch'] = 'rest'
                n_v.loc[:, 'Octave'] = 'rest'
                n_v.loc[:, 'MIDI'] = np.nan
                n_v.loc[:, 'Onset'] = 0.0
                n_v.loc[:, 'Duration'] = ub_r_np[i][2]
                n_v.loc[:, 'Upbeat'] = 'true'
                df = _inseart_row_in_pd(row_number=m_idx, df=df, row_value=n_v)

            self.compute_measure_onset(df)
            df = self._re_compute_onset_upbeat(df)
        return df

    def upbeat_detection(self,df_c):
        ROUND = 2
        df_c['Upbeat'] = ['none' for _ in range(len(df_c))]
        df_c['MeasureDurrDiff'] = [0 for _ in range(len(df_c))]


        u_parts = np.unique(np.squeeze(df_c[['PartID']].to_numpy()))
        for c_p in u_parts:
            grouped = df_c.groupby(df_c.PartID)
            try:
                p_all = grouped.get_group(int(c_p)).copy()
            except:
                p_all = grouped.get_group(str(c_p)).copy()
            p_all = p_all.drop_duplicates(subset='Onset', keep='last')
            gp_tt = p_all[['Measure', 'TieType']].drop_duplicates()
            gp_tt = gp_tt[gp_tt['TieType'] == 'start']['Measure'].to_numpy()

            gp_d1 = p_all.groupby(['Onset', 'Measure', 'PartID', 'MeasureDuration'], as_index=False)['Duration'].sum()

            gp_d1 = gp_d1.groupby(['Measure', 'PartID', 'MeasureDuration'], as_index=False)['Duration'].sum()
            gp_d1.Duration = gp_d1.Duration.round(ROUND)

            gp_d1["ck_upbeat"]  = np.where((gp_d1["MeasureDuration"] == gp_d1["Duration"]),  False, True)

            gp_d1['MeasureDurrDiff'] = gp_d1['MeasureDuration'] - gp_d1['Duration']
            gp_d1.MeasureDurrDiff = gp_d1.MeasureDurrDiff.round(ROUND)
            # gp_d1.Onset = gp_d1.Onset.round(ROUND)

            gp_d2 = gp_d1[['Measure', 'PartID', 'ck_upbeat', 'MeasureDuration', 'MeasureDurrDiff']].copy()
            if False:
                # Setting tied notes to False
                for tt in gp_tt:
                    curr_dur_diff = gp_d2.loc[(gp_d2["Measure"] == tt),"MeasureDurrDiff"].to_numpy()[0]
                    curr_mea_dur = gp_d2.loc[(gp_d2["Measure"] == tt),"MeasureDuration"].to_numpy()[0]
                    if curr_dur_diff <0:
                        curr_mea_dur = abs(curr_mea_dur)
                        curr_dur_diff = abs(curr_dur_diff)
                        if curr_dur_diff>curr_mea_dur:
                            r = curr_dur_diff%curr_mea_dur
                            if r == 0:
                                gp_d2.loc[(gp_d2["Measure"] == tt), "ck_upbeat"] = False
                            else:
                                pass
                    tt_n = tt.copy()
                    for i in range(tt_n, tt_n+1):
                        tt_n +=1
                        print("testing -- ",tt_n)

                        try:
                            if curr_dur_diff == gp_d2.loc[(gp_d2["Measure"] == tt_n), "MeasureDurrDiff"].to_numpy()[0]:
                                gp_d2.loc[(gp_d2["Measure"] == tt_n), "ck_upbeat"] = False
                                break
                        except:
                            pass
                    gp_d2.loc[(gp_d2["Measure"] == tt),"ck_upbeat"] = False

            gp_d2 = gp_d2.to_numpy()
            for i in gp_d2:
                df_c.loc[(df_c["Measure"] == i[0]) & (df_c["PartID"] == i[1]),"Upbeat"] = i[2]
                df_c.loc[(df_c["Measure"] == i[0]) & (df_c["PartID"] == i[1]),"MeasureDurrDiff"] = i[4]
        return df_c

    def upbeat_correction(self, df_c):
        # ck_ = set(df_c['Upbeat'].tolist())
        u_parts = np.unique(np.squeeze(df_c[['PartID']].to_numpy()))

        for c_p in u_parts:
            grouped = df_c.groupby(df_c.PartID)
            try:
                p1 = grouped.get_group(int(c_p)).copy()
            except:
                p1 = grouped.get_group(str(c_p)).copy()
            measure_info = p1['Measure'].to_numpy(np.int)

            first_mearsure = min(measure_info)
            last_mearsure = max(measure_info)
            gp1 = p1[['Measure', 'PartID', 'MeasureDuration', 'MeasureDurrDiff', 'Upbeat']]
            gp1 = gp1.loc[(gp1["Upbeat"] == True)].drop_duplicates()
            gp1['rest_up'] = gp1['MeasureDuration'] - gp1['MeasureDurrDiff']
            gp1.reset_index(inplace=True)
            for index, row in gp1.iterrows():
                self.onset_rin = row['MeasureDurrDiff']
                self.onset_rin_prev = row['MeasureDurrDiff']
                if row['Measure'] in [first_mearsure, last_mearsure]:
                    if row['Measure'] == first_mearsure:
                        copy_idx = min(df_c.index[df_c['PartID'] == row['PartID']])
                        uppend_idx = copy_idx

                    elif row['Measure'] == last_mearsure:
                        copy_idx = max(df_c.index[df_c['PartID'] == row['PartID']])
                        uppend_idx = copy_idx+1

                    n_v = df_c.iloc[[copy_idx]].copy()
                    n_v.loc[:, '#Note_Debug'] = 'added_UB'
                    n_v.loc[:, 'Pitch'] = 'rest'
                    n_v.loc[:, 'Octave'] = 'rest'
                    n_v.loc[:, 'MIDI'] = np.nan
                    if row['Measure'] == first_mearsure:
                        n_v.loc[:, 'Onset'] = 0.0
                        df_z = df_c.loc[ (df_c["Measure"] == row['Measure']) & (df_c["PartID"] == row['PartID'])]

                        for r_idx, r_i in df_z.iterrows():

                            if r_i['ChordTag'] == 'chord':
                                # preserving last onset_value if there is chord.
                                # TODO: Check for trid and four chord voice.
                                df_c.loc[r_idx, 'Onset']= self.onset_rin_prev
                            else:
                                df_c.loc[r_idx, 'Onset']= self.onset_rin
                                df_c.loc[r_idx, 'LocalOnset']= self.onset_rin
                                self.onset_rin += r_i['Duration']
                    else:
                        # To find the end of the last note n_v['Onset']+n_v['Duration']
                        # TODO: Test for chords
                        n_v.loc[:, 'Onset'] = n_v['Onset']+n_v['Duration']
                        n_v.loc[:, 'LocalOnset'] = abs(n_v['MeasureOnset'] - n_v['Onset'])


                    n_v.loc[:, 'Duration'] = row['MeasureDurrDiff']
                    n_v.loc[:, 'MeasureDurrDiff'] = np.nan
                    n_v.loc[:, 'Upbeat'] = True
                    n_v.loc[:, 'ChordTag'] = 'none'
                    n_v.loc[:, 'TieType'] = 'none'
                    n_v.loc[:, 'GraceTag'] = 'none'
                    df_c = _inseart_row_in_pd(row_number=uppend_idx, df=df_c, row_value=n_v)
                    # p_all = _inseart_row_in_pd(row_number=uppend_idx, df=p_all, row_value=n_v, reset_index=True)
                else:pass

        for c_p2 in u_parts:
            # print("------------------------")
            grouped = df_c.groupby(df_c.PartID)
            try:
                p2 = grouped.get_group(int(c_p2)).copy()
            except:
                p2 = grouped.get_group(str(c_p2)).copy()
            measure_info = p2['Measure'].to_numpy(np.int)
            first_mearsure = min(measure_info)
            last_mearsure = max(measure_info)
            min_idx_parts = min(list(p2.index.values))
            max_idx_parts = max(list(p2.index.values))

            gp11 = p2[['Measure', 'PartID', 'MeasureDuration', 'MeasureDurrDiff', 'Upbeat']]
            gp11 = gp11.loc[(gp11["Upbeat"] == True)].drop_duplicates()
            gp11['rest_up'] = gp11['MeasureDuration'] - gp11['MeasureDurrDiff']
            gp11.reset_index(inplace=True)

            for index, row in gp11.iterrows():
                # TODO: First find if there is upbeat in the seccussive measure
                if row['Measure'] in [first_mearsure, last_mearsure]:
                    continue
                else:

                    gp22 = gp11.iloc[index:index+2]
                    if len(gp22) == 2:
                        assert len(gp22) == 2, f"Please check Index, only two should be selected to find consecutive measure for summing \n{gp22} Last measure\n{last_mearsure} \n{p2}"
                        # to check if there are consicutive measure add up ???
                        m_ck = gp22['Measure'].to_numpy()
                        assert len(m_ck) == 2, f"Please check Index, only two should be selected to find consecutive measure for summing\n{m_ck}"
                        # check for consecutive measure
                        if m_ck[0]+1 == m_ck[1]:
                            u = np.unique(gp22['MeasureDuration'].to_numpy())
                            assert len(u) == 1, f"Problem Detecting Upbeat or special case to be implemented - Please use " \
                                         f"another xml files. "
                            d_u = np.squeeze(u)
                            md_sum = gp22['MeasureDurrDiff'].to_numpy().sum()
                            # If the sum of two consicutive measure is qual to the total duration then merge them into one measure - Upbeat=True
                            if d_u == md_sum:
                                #TWO  consecutive measure with upbeat detected
                                index_to_reset = p2[p2['Measure'] == m_ck[1]].index[0]
                                # print(df_c)

                                df_c = self._recompute_df_upbeat_mid(ub_st_idx=min_idx_parts,
                                                              ub_en_idx=max_idx_parts, index_to_reset=index_to_reset,
                                                              df_c=df_c)

                            # if np.sum(gp2['MeasureDurrDiff'].to_numpy()) == u[0]:
                            #     print("UPBEAT DETECTED In the middle of the measure -  TO BE IMPLEMENTED")

                        else:
                            pass
                    else:

                        pass
                    # TODO:
                    # 1. Cross check notes if they are less than measure
                    # 2. Cross check notes if they are longer than measure + tied values should be "start"
                    # # except:
                    #     pass

        return df_c
    def compute_local_onset(self, df_c):
        ROUND = 2
        df_c['LocalOnset'] = pd.to_numeric(df_c['Onset']) - pd.to_numeric(df_c['MeasureOnset'])

        df_c.LocalOnset = df_c.LocalOnset.round(ROUND)
        return df_c

class XMLParser(XMLToolBox):
    def __init__(self, path, ignore_upbeat, ignore_ties, *args, **kwargs):
        super().__init__(path,ignore_upbeat, ignore_ties, *args, **kwargs)
        self.path = path
        self.logger = kwargs['logger']
        self.ignore_upbeat = ignore_ties
        self.ignore_ties = ignore_ties

    def xml_parse(self, *args, **kwargs):
        if isfile(self.path) == False:
            self.logger.info(f"File Not Found - {self.path}")
            raise FileExistsError("File Does Note Exist".format(self.path))
        else:
            self.logger.info(f"File Found - {self.path}")

        df_data = self.strip_xml()
        df_data = self.compute_measure_onset(df_data)
        df_data = self.compute_voice_onset(df_data)
        df_data = self.convert_pitch_midi(df_data)
        df_data = self.compute_local_onset(df_data)
        if not self.ignore_upbeat:
            df_data = self.upbeat_detection(df_data)
            df_data = self.upbeat_correction(df_data)
        if not self.ignore_ties:
            df_data = self.compute_tie_duration(df_data)
        df_data = self.remove_df_cols(df_data)


        return df_data