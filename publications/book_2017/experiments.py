import pandas as pd
import os
import numpy as np
import time
import seaborn as sns
import pandas as pn
import matplotlib.pyplot as pl
sns.set(style="white", color_codes=True)
from scipy.stats import ttest_ind, ttest_rel, pearsonr, linregress
from tools import AudioAnalysisTools, TextWriter

__author__ = 'Jakob Abesser'


class AudioAnalysisExperiments:

    def __init__(self,
                 dir_data,
                 dir_results,
                 min_num_solos_per_artist=4,
                 debug=True):
        self.dir_results = dir_results
        self.dir_data = dir_data
        self.debug = debug

        # data loading & preparation
        self.df_notes, self.df_beats, self.df_solos = self.load_data()
        self.mel_ids = np.array(self.df_solos.index)
        self.num_solos = len(self.mel_ids)
        self.df_performer, self.df_performer_all = self.create_performer_data_frame(min_num_solos_per_artist=min_num_solos_per_artist)

        self.tools = AudioAnalysisTools
        self.extractors = {#'metadata': [self.metadata_analysis],
                           #'tuning': [self.tuning_analysis_tuning_freq_vs_recording_year],
                           'intensity': [#self.intensity_analysis_correlations,
                                         #self.intensity_analysis_eight_note_sequences
                                         self.intensity_phrase_contours]}
        self.text_writer = TextWriter()

    def load_data(self):
        return [pd.read_pickle(os.path.join(self.dir_data, 'df_%s.dat' % _)) for _ in ['notes', 'beats', 'solos']]

    def run(self):
        for category in self.extractors.keys():
            for extractor in self.extractors[category]:
                extractor()
        print('Finished all experiments :)')

    def metadata_analysis(self):
        self.text_writer.reset()
        for i, _ in enumerate(self.df_performer.index):
            self.text_writer.add(self.df_performer.index[i] +
                            " & " +
                            ", ".join(self.df_performer['instrument'][i]) +
                            " & " + str(self.df_performer['num_solos'][i]) +
                            " & " + str(self.df_performer['num_notes'][i]) + '\\\\')
        # add additional line with total number of solos and notes
        self.text_writer.add("\hline")
        self.text_writer.add("\\textbf{Total} & & \\textbf{" + \
                        str(np.sum(self.df_performer['num_solos'])) + "} & \\textbf{" + \
                        str(np.sum(self.df_performer['num_notes'])) + "} \\\\")
        self.text_writer.save(os.path.join(self.dir_results,
                                      'metadata_analysis_artist_stats.txt'))

    def tuning_analysis_tuning_freq_vs_recording_year(self):
        delta_f_ref_440_cent = np.abs(1200.*np.log2(self.df_solos['tuning_frequency']/440.))
        year = self.df_solos['recording_year']

        # linear regression
        slope, intercept, r_value, p_value, std_err = linregress(year, delta_f_ref_440_cent)

        # write r & p values to files
        self.text_writer.reset()
        self.text_writer.add("%.3f" % r_value)
        self.text_writer.save(os.path.join(self.dir_results, 'tuning_analysis_tuning_freq_vs_recording_year_lin_reg_r.txt'))
        self.text_writer.reset()
        self.text_writer.add(self.tools.generate_p_value_string(p_value))
        self.text_writer.save(os.path.join(self.dir_results, 'tuning_analysis_tuning_freq_vs_recording_year_lin_reg_p.txt'))


        # scatter plot with linear regression line
        df = pn.DataFrame({'Year': year,
                           '$\left|\Delta f_\mathrm{ref}^\mathrm{440}\\right|$ [cent]': delta_f_ref_440_cent})
        sns.set_style("whitegrid")
        # sns.set(font_scale=0.7)
        g = sns.lmplot(x='Year',
                       y='$\left|\Delta f_\mathrm{ref}^\mathrm{440}\\right|$ [cent]',
                       markers="o",
                       data=df, ci=None, palette="Blues_d", size=3,
                       scatter_kws={"s": 10, "alpha": 1})
        g.set_xticklabels(rotation=60)
        g.set(ylim=(0, 50))
        pl.savefig(os.path.join(self.dir_results, 'tuning_analysis_tuning_freq_vs_recording_year_scatterplot.eps'),
                   bbox_inches='tight')
        pl.close()

    def intensity_phrase_contours(self, min_phrase_len=4):

        print("Experiment: Phrase contours: intensity vs. pitch")

        contour_type = ('', 'horizontal', 'convex', 'concave', 'ascending', 'descending')
        contour_type_label = ('$< {}$ notes'.format(min_phrase_len), 'Horizontal', 'Convex', 'Concave', 'Ascending', 'Descending')
        contour_type_label_short = ('$< {}$ notes'.format(min_phrase_len), 'Horiz.', 'Conv.', 'Conc.', 'Ascend.', 'Descend.')
        num_contour_type = len(contour_type)
        num_contour_type_loud = np.zeros((self.num_solos, num_contour_type))
        num_contour_type_pitch = np.zeros((self.num_solos, num_contour_type))
        num_contour_type_joint = np.zeros((num_contour_type, num_contour_type))

        # get raw note parameters
        phrase_id_values = self.df_notes['phraseid'].as_matrix()
        intensity_values = self.df_notes['intensity_solo_median'].as_matrix()
        pitch_values = self.df_notes['pitch'].as_matrix()

        for s, mel_id in enumerate(self.mel_ids):
            print('Solo {} / {}'.format(s + 1, self.num_solos))

            note_idx = np.where(self.df_notes['melid'] == mel_id)[0]
            phrase_id = phrase_id_values[note_idx]
            num_phrases = max(phrase_id)
            num_notes = len(phrase_id)

            for p in range(num_phrases):
                # notes in current phrase
                note_id_in_phrase = np.where(phrase_id == p + 1)[0]

                # intensity contour
                contour_type_loud_idx = 0  # '' = default contour type
                if len(note_id_in_phrase) > min_phrase_len:
                    contour_type_loud = self.tools.compute_abesser_contour(intensity_values[note_idx[note_id_in_phrase]])
                    contour_type_loud_idx = contour_type.index(contour_type_loud)

                num_contour_type_loud[s,
                                      contour_type_loud_idx] += 1

                # pitch contour
                contour_type_pitch_idx = 0  # '' = default contour type
                if len(note_id_in_phrase) > min_phrase_len:
                    contour_type_pitch = self.tools.compute_abesser_contour(pitch_values[note_idx[note_id_in_phrase]])
                    contour_type_pitch_idx = contour_type.index(contour_type_pitch)

                num_contour_type_pitch[s,
                                       contour_type_pitch_idx] += 1

                # count co-occurrances of both contour types
                num_contour_type_joint[contour_type_pitch_idx,
                                          contour_type_loud_idx] += 1

        # normalize to %
        num_contour_type_loud = np.sum(num_contour_type_loud, axis=0)
        num_contour_type_loud /= np.sum(num_contour_type_loud)
        num_contour_type_loud *= 100.
        num_contour_type_pitch = np.sum(num_contour_type_pitch, axis=0)
        num_contour_type_pitch /= np.sum(num_contour_type_pitch)
        num_contour_type_pitch *= 100.

        # focus on contours over critical length
        num_contour_type_joint = num_contour_type_joint[1:, 1:]
        num_contour_type_joint /= np.tile(np.sum(num_contour_type_joint, axis=1).reshape(-1, 1),
                                             (1, num_contour_type - 1)) # TODO simplify using broadcasting
        num_contour_type_joint *= 100.

        # export results for latex table
        self.text_writer.reset()
        for _ in range(num_contour_type):
            self.text_writer.add('{} & {:2.1f} & {:2.1f} \\\\'.format(contour_type_label[_],
                                                                      num_contour_type_loud[_],
                                                                      num_contour_type_pitch[_]))
        self.text_writer.save(os.path.join(self.dir_results, 'intensity_analysis_contour_pitch_intensity.txt'))

        self.text_writer.reset()
        self.text_writer.add(" & " + " & ".join(contour_type_label_short[1:]) + "\\\\")
        for row in range(num_contour_type - 1):
            s = " & ".join(
                [contour_type_label[row + 1]] + ["{:2.1f}".format(num_contour_type_joint[row, col]) for col in
                                                 range(num_contour_type - 1)])
            s += "\\\\"
            self.text_writer.add(s)

        fn_result = os.path.join(self.dir_results, 'intensity_analysis_contour_joint.txt')
        self.text_writer.save(fn_result)
        print('{} saved ...'.format(fn_result))


    def intensity_analysis_eight_note_sequences(self,
                                                min_num_notes=10,
                                                num_examples_table=10):
        """ Check which solos show significant louder notes on first or second eight notes
        Args:
            min_num_notes (int): Minimum length of eight-note sequences to be considered
            num_examples_table (int): Overall number of examples to show in result table
        """
        print("Experiment: Loudness differences between first and second eight notes")

        all_p = []
        all_d = []
        all_n = []

        # get median intensity values for all notes
        intensity_values = self.df_notes['intensity_solo_median'].as_matrix()

        # iterate over solos
        for sid, mel_id in enumerate(self.mel_ids):

            print('Solo {} / {}'.format(sid + 1, self.num_solos))

            # find all notes of current solo in df_notes
            note_idx = np.where(self.df_notes['melid'] == mel_id)[0]
            num_notes = len(note_idx)

            # load metrical positions of notes in current solo
            division = np.zeros(num_notes, dtype=int)
            bar_number = np.zeros(num_notes, dtype=int)
            beat_number = np.zeros(num_notes, dtype=int)
            tatum_number = np.zeros(num_notes, dtype=int)
            for nid in range(num_notes):
                division[nid] = self.df_notes['division'][note_idx[nid]]
                bar_number[nid] = self.df_notes['bar'][note_idx[nid]]
                beat_number[nid] = self.df_notes['beat'][note_idx[nid]]
                tatum_number[nid] = self.df_notes['tatum'][note_idx[nid]]

            intensity_first_eight = []
            intensity_second_eight = []

            # iterate over all bars
            for bar_id in np.unique(bar_number):

                # iterate over all beats within current bar, where eight notes exist (equals beat-division of 2)
                for beat_id in np.unique(beat_number[np.logical_and(bar_number == bar_id,
                                                                    division == 2)]):

                    # get note index of current eight-note-pair
                    note_id_cand = np.where(np.logical_and(bar_number == bar_id,
                                                           beat_number == beat_id))[0]
                    # corresponding tatum values
                    tatum_cand = tatum_number[note_id_cand]

                    # check if we have 2 successive eight notes
                    if 1 in tatum_cand and 2 in tatum_cand:
                        # note indices of first and second eight note
                        note_idx_first_eight_note = note_idx[note_id_cand[tatum_cand == 1][0]]
                        note_idx_second_eight_note = note_idx[note_id_cand[tatum_cand == 2][0]]

                        # save corresponding intensity values
                        intensity_first_eight.append(intensity_values[note_idx_first_eight_note])
                        intensity_second_eight.append(intensity_values[note_idx_second_eight_note])
    #
            intensity_first_eight = np.array(intensity_first_eight)
            intensity_second_eight = np.array(intensity_second_eight)

            # paired t-test (compute difference between groups and run one-sample t test)
            t, p = ttest_rel(intensity_first_eight,
                             intensity_second_eight)

            # cohen's d (effect size measure for paired t-test) -> 0.2 (small), 0.5 (medium), 0.8 (large)
            d = self.tools.cohens_d(intensity_first_eight,
                                    intensity_second_eight)

            # store results of t-test
            all_p.append(p)  # significance level
            all_d.append(d)  # signed effect size
            all_n.append(len(intensity_first_eight))  # number of eight-note-pairs in solo

        all_d = np.array(all_d)
        all_p = np.array(all_p)
        all_n = np.array(all_n)

        # select solos with
        #  - significant difference between first and second eight notes &
        #  - minimum of 10 eight-note pairs
        idx_select = np.where(np.logical_and(all_p < 0.05,
                                             all_n > min_num_notes))[0]

        print("{} solos with positive d, {} solos with negative d, total = {}".format(np.sum(all_d[idx_select] > 0),
                                                                                      np.sum(all_d[idx_select] < 0),
                                                                                      len(all_d)))

        # create table with the N solos of both categories with the highest absolute effect size
        self.text_writer.reset()
        idx_pos = (all_d[idx_select] > 0).nonzero()[0]
        idx_neg = (all_d[idx_select] < 0).nonzero()[0]

        # iterate over solos with positive and negative effect size
        for k, idx_curr_category in enumerate((idx_pos, idx_neg)):
            solo_idx_curr_category = idx_select[idx_curr_category]

            # sort solos of current selection (pos. or neg. d values) in descending order based on absolute value
            sort_idx = np.argsort(np.abs(all_d[solo_idx_curr_category]))

            # flip sort order to descending for solos with positive d
            if k == 0:
                sort_idx = sort_idx[::-1]

            # solo ids in sorted order
            solo_idx_curr_category = solo_idx_curr_category[sort_idx]

            # create row entries in table
            for _ in range(num_examples_table):

                # avoid overflow
                if _ < len(sort_idx):

                    # write solo metadata into row
                    solo_id = solo_idx_curr_category[_]
                    self.text_writer.add("{} & {} & {:2.1f} & {} \\\\".format(self.get_artist_instrument_label(self.df_solos['performer'][self.mel_ids[solo_id]]),
                                                                              self.df_solos['title'][self.mel_ids[solo_id]],
                                                                              all_d[solo_id],
                                                                              self.tools.generate_p_value_string(all_p[solo_id])))
            if k == 0:
                self.text_writer.add("\hline")

        fn_result = os.path.join(self.dir_results, 'intensity_analysis_first_second_eights.txt')
        self.text_writer.save(fn_result)
        print('{} saved ...'.format(fn_result))

        # export all solo-wise results
        self.text_writer.reset()
        for idx in range(len(self.mel_ids)):
            self.text_writer.add("{} & {} & {:2.1f} & {} \\\\".format(self.get_artist_instrument_label(self.df_solos['performer'][self.mel_ids[idx]]),
                                                                      self.df_solos['title'][self.mel_ids[idx]],
                                                                      all_d[idx],
                                                                      self.tools.generate_p_value_string(all_p[idx])))
        fn_result = os.path.join(self.dir_results, 'intensity_analysis_first_second_eights_all.txt')
        self.text_writer.save(fn_result)

    def intensity_analysis_correlations(self, num_highest_and_lowest_artists=7):

        param_labels = ('Pitch', 'Duration', 'RelPosInPhrase')
        num_labels = len(param_labels)
        r = np.zeros((num_labels, self.num_solos))
        p = np.zeros((num_labels, self.num_solos))

        phrase_id = self.df_notes['phraseid'].as_matrix()
        intensity = self.df_notes['intensity_solo_median'].as_matrix()
        pitch = self.df_notes['pitch'].as_matrix()
        duration = self.df_notes['duration'].as_matrix()

        # iterate over solos
        for sid, melid in enumerate(self.mel_ids):

            # get note indices of current solo with valid intensity values
            note_idx = np.where(self.df_notes['melid'] == melid)[0]

            # get note parameters
            curr_intensity = intensity[note_idx]
            curr_duration = duration[note_idx]
            curr_pitch = pitch[note_idx]
            curr_phrase_id = phrase_id[note_idx]
            assert all(np.diff(curr_phrase_id) >= 0)
            curr_rel_pos_in_phrase = self.tools.get_relative_position_in_phrase(curr_phrase_id)

            vec = (curr_pitch, curr_duration, curr_rel_pos_in_phrase)
            N = len(vec)
            # compute Pearson correlation coefficient
            for pid in range(3):
                r[pid, sid], p[pid, sid] = pearsonr(curr_intensity, vec[pid])

        # average r over solos for each artist with significant correlations
        artist = np.array(self.df_solos['performer'].as_matrix())
        # focus on artist selection
        unique_artist = np.array(self.df_performer.index)
        unique_artist_label = np.array(self.df_performer['artist_instrument_label'])
        num_unique_artist = len(unique_artist)

        r_artist_mean = -1*np.ones((num_labels, num_unique_artist))
        r_artist_std = -1*np.ones((num_labels, num_unique_artist))
        r_artist_valid = np.ones((num_labels, num_unique_artist), dtype=bool)

        for aid, a in enumerate(unique_artist):
            solo_idx = np.where(artist == a)[0]

            for lid in range(num_labels):
                # focus on significant correlations
                solo_idx = solo_idx[p[lid, solo_idx] < .05]

                if len(solo_idx) > 1:
                    r_artist_mean[lid, aid] = np.mean(r[lid, solo_idx])
                    r_artist_std[lid, aid] = np.std(r[lid, solo_idx])
                else:
                    r_artist_valid[lid, aid] = False

        # get matrix with row-wise solo idx in descending order of correlation coefficient size
        sorted_indices = [[] for _ in range(3)]

        # sorted_indices = np.zeros((num_labels, num_unique_artist), dtype=int)
        for lid in range(num_labels):
            # valid artists with significant correlations
            valid_artist_idx = np.where(r_artist_valid[lid, :])[0]
            sorted_indices[lid] = valid_artist_idx[np.argsort(r_artist_mean[lid, valid_artist_idx])[::-1]]
            print(np.min(r_artist_mean[lid, valid_artist_idx]))

        ranks = [None for _ in range(num_labels)]
        for lid in range(num_labels):
            num_valid_artists = len(sorted_indices[lid])
            ranks[lid] = np.concatenate((np.arange(num_highest_and_lowest_artists),
                                         np.arange(num_valid_artists - num_highest_and_lowest_artists, num_valid_artists)))

        all_ranks = np.concatenate((np.arange(num_highest_and_lowest_artists),
                                    np.arange(num_unique_artist - num_highest_and_lowest_artists,
                                              num_unique_artist)))

        self.text_writer.reset()
        for _, rank in enumerate(all_ranks):

            self.text_writer.add(
                "{} & {} \\newline ({:1.2f}~$\pm$~{:1.2f}) & {} \\newline ({:1.2f}~$\pm$~{:1.2f}) & {} \\newline ({:1.2f}~$\pm$~{:1.2f}) \\\\".format(
                    rank + 1,
                    unique_artist_label[sorted_indices[0][ranks[0][_]]],
                    r_artist_mean[0, sorted_indices[0][ranks[0][_]]],
                    r_artist_std[0, sorted_indices[0][ranks[0][_]]],
                    unique_artist_label[sorted_indices[1][ranks[1][_]]],
                    r_artist_mean[1, sorted_indices[1][ranks[1][_]]],
                    r_artist_std[1, sorted_indices[1][ranks[1][_]]],
                    unique_artist_label[sorted_indices[2][ranks[2][_]]],
                    r_artist_mean[2, sorted_indices[2][ranks[2][_]]],
                    r_artist_std[2, sorted_indices[2][ranks[2][_]]]))
            if _ == num_highest_and_lowest_artists - 1:
                self.text_writer.add('\hline')
        fn_result = os.path.join(self.dir_results, 'intensity_analysis_correlations.txt')
        self.text_writer.save(fn_result)
        print('{} saved ...'.format(fn_result))

    def create_performer_data_frame(self, min_num_solos_per_artist=4):
        unique_performer = np.unique(self.df_solos['performer'])
        num_unique_performer = len(unique_performer)
        unique_performer_inst = ['' for _ in unique_performer]
        unique_performer_num_solos = np.zeros(num_unique_performer, dtype=int)
        unique_performer_num_notes = np.zeros(num_unique_performer, dtype=int)
        # get instrument(s), number of solos, number of notes for each unique performer
        for id, up in enumerate(unique_performer):
            unique_performer_inst[id] = sorted(list(set(self.df_solos['instrument'][self.df_solos['performer'] == up])))
            unique_performer_num_solos[id] = np.sum(self.df_solos['performer'] == up)
            melids = self.df_solos['melid'][self.df_solos['performer'] == up]
            for melid in melids:
                unique_performer_num_notes[id] += len(self.df_notes['onset'][self.df_notes['melid'] == melid])

        # collect results in pandas dataframe
        df_performer = pd.DataFrame({'instrument': unique_performer_inst,
                                     'num_notes': unique_performer_num_notes,
                                     'num_solos': unique_performer_num_solos,
                                     'artist_instrument_label': ["%s (%s)" % (unique_performer[_], ', '.join(unique_performer_inst[_])) for _ in range(num_unique_performer)]},
                                    index=unique_performer)

        df_performer_reduced = df_performer.ix[df_performer['num_solos'] >= min_num_solos_per_artist]

        return df_performer_reduced, df_performer

    def get_artist_instrument_label(self, performer):
        return self.df_performer_all['artist_instrument_label'][self.df_performer_all.index == str(performer)][0]