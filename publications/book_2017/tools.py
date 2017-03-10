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
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
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
        num_notes_per_phrase_id = np.zeros(num_phrases, dcontour_type=float)
        note_count_per_phrase = np.zeros(num_phrases, dcontour_type=float)
        rel_pos_in_phrase = np.zeros(num_notes, dcontour_type=float)

        # get number of notes in each phrase
        for pid, curr_phrase_id in enumerate(unique_phrase_id):
            num_notes_per_phrase_id[pid] = np.sum(phrase_id == curr_phrase_id)

        # compute relative in-phrase-position for each note (0 - first note, 1 - last note)
        for nid, curr_phrase_id in enumerate(phrase_id):
            curr_phrase_num = np.where(unique_phrase_id == curr_phrase_id)[0][0]
            rel_pos_in_phrase[nid] = note_count_per_phrase[curr_phrase_num] / (num_notes_per_phrase_id[curr_phrase_num] - 1)
            note_count_per_phrase[curr_phrase_num] += 1.

        return rel_pos_in_phrase


    @staticmethod
    def cohens_d(x, y):
        """ Measure Cohen's d, which is an effect size measure for paired t-test
           -> 0.2 (small), 0.5 (medium), 0.8 (large)
           """
        return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)

    @staticmethod
    def compute_abesser_contour(vals, 
                                segment_size_percent=None,
                                diff_abs_tolerance_rel=.1):
        """ Compute contour shape based on variation of Huron contour
            as proposed in 
            [1] Abesser, J., Frieler, K., Cano, E., Pfleiderer, M., & 
            Zaddach, W.-G. (2016). Score-Informed Analysis of Tuning, 
            Intonation, Pitch Modulation, and Dynamics in Jazz Solos. 
            IEEE/ACM Transactions on Audio, Speech, and Language 
            Processing,
        Args:
            vals (ndarray): Note-wise values (e.g. pitch / intensity) within
                            structural segment (e.g. phrase)
            segment_size_percent (None / ndarray): Relative segment size (3 segments!)
            diff_abs_tolerance_rel (float): Tolerance w.r.t. absolute deviation
                                            of median values in adjacent segments
                                            (relative to overall range of values)
        Returns:
            contour_type (string): Contour type ('horizontal', 'convex', 'concave',
                                   'ascending', 'descending')
        """
        if segment_size_percent is None:
            segment_size_percent = np.array([.25, .5, .25])
        assert len(segment_size_percent) == 3
        
        diff_abs_tolerance = diff_abs_tolerance_rel*(np.max(vals) - np.min(vals))
        num_values = len(vals)
        borders = np.round(np.cumsum(segment_size_percent)[:-1]*num_values)
        borders = np.insert(borders, 0, 0)
        borders = np.append(borders, num_values)
        num_segments = len(segment_size_percent)
        segment_vals = np.zeros(num_segments)
        for s in range(num_segments):
            segment_vals[s] = np.median(vals[borders[s]:borders[s+1]])
        comp = [0, 0]
        for _ in range(2):
            if abs(segment_vals[_] - segment_vals[_+1]) > diff_abs_tolerance:
                if segment_vals[_] > segment_vals[_+1]:
                    comp[_] = -1
                else:
                    comp[_] = 1
        contour_type = ''
        if comp == [0, 0]:
            contour_type = 'horizontal'
        elif comp == [1, -1]:
            contour_type = 'convex'
        elif comp == [-1, 1]:
            contour_type = 'concave'
        elif comp == [0, 1] or comp == [1, 1] or comp == [1, 0]:
            contour_type = 'ascending'
        elif comp == [0, -1] or comp == [-1, -1] or comp == [-1, 0]:
            contour_type = 'descending'
        return contour_type
        

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