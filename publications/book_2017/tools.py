import numpy as np

__author__ = 'Jakob Abesser'


class AudioAnalysisTools:

    @staticmethod
    def write_to_text_file(fn_txt, num, is_prob=False):
        if is_prob:
            content = AudioAnalysisTools.generate_p_value_string(num)
        else:
            content = str(num)

        with open(fn_txt, 'w+') as f:
            f.write(content)

    @staticmethod
    def generate_p_value_string(p):
        if p < 0.001:
            return 'p < 0.001'
        elif p < 0.01:
            return 'p < 0.01'
        elif p < 0.05:
            return 'p < 0.05'
        else:
            return 'n. s.'

    @staticmethod
    def get_relative_position_in_phrase(phrase_id):
        """ Based on order of notes in the same phrase, this function derives a relative
            position of each note in its phrase
        Args:
            phrase_id (ndarray): Note-wise phrase ID values
        Returns:
            rel_pos_in_phrase (ndarray): Note-wise relative position within corresponding phrase
        """
        num_notes = len(phrase_id)
        unique_phrase_id = np.unique(phrase_id)
        num_phrases = len(unique_phrase_id)

        # initialize
        num_notes_per_phrase_id = np.zeros(num_phrases, dtype=float)
        note_count_per_phrase = np.zeros(num_phrases, dtype=float)
        rel_pos_in_phrase = np.zeros(num_notes, dtype=float)

        # get number of notes in each phrase
        for pid, curr_phrase_id in enumerate(unique_phrase_id):
            num_notes_per_phrase_id[pid] = np.sum(phrase_id == curr_phrase_id)

        # compute relative in-phrase-position for each note (0 - first note, 1 - last note)
        for nid, curr_phrase_id in enumerate(phrase_id):
            curr_phrase_num = np.where(unique_phrase_id == curr_phrase_id)[0][0]
            rel_pos_in_phrase[nid] = note_count_per_phrase[curr_phrase_num] / (num_notes_per_phrase_id[curr_phrase_num] - 1)
            note_count_per_phrase[curr_phrase_num] += 1.

        return rel_pos_in_phrase



class TextWriter:
    def __init__(self):
        self.content = None
        self.reset()

    def reset(self):
        self.content = []

    def add(self, val):
        self.content.append(val)

    def save(self, fnTXT):
        with open(fnTXT, 'w+') as f:
            for _ in self.content:
                f.write(_ + '\n')