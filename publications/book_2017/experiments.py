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
                 min_num_solos_per_artist=3):
        self.dir_results = dir_results
        self.dir_data = dir_data

        # data loading & preparation
        self.df_notes, self.df_beats, self.df_solos = self.load_data()
        self.df_performer = self.create_performer_data_frame(min_num_solos_per_artist=min_num_solos_per_artist)

        self.tools = AudioAnalysisTools
        self.extractors = {'metadata': [self.metadata_analysis],
                           'tuning': [self.tuning_analysis_tuning_freq_vs_recording_year],
                           'intensity': [self.intensity_analysis_correlations,
                                         self.intensity_analysis_eight_note_sequences]}
        self.unique_melids = np.unique(self.df_solos['melid'])
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

    def intensity_analysis_eight_note_sequences(self, min_num_notes=10, num_examples_table=10):
        """ Check which solos show significant louder notes on first or second eight notes """
        print("Experiment: Loudness differences between first and second eight notes")
        num_solos = self.df_solos.shape[0]

        all_p = []
        all_d = []
        all_n = []

        # get peak intensity values for all notes
        intensity_values = self.df_notes['intensity_solo_max'].as_matrix()

        # iterate over solos
        for sid in range(num_solos):

            print('Solo {} / {}'.format(sid + 1, num_solos))

            # find all notes of current solo in df_notes
            base_name = self.df_solos['base_name'][sid]
            note_idx = np.where(self.df_notes['base_name'] == base_name)[0]

            # make sure onsets are in ascending order
            note_idx = np.sort(note_idx)
            onsets = self.df_notes['onset'].as_matrix()[note_idx]
            assert all(np.diff(onsets) > 0)
            num_notes = len(note_idx)

            # matrix to store division, bar_num, beat_num, tatum_num
            met_pos_mat = []

            # get metrical positions of notes in current solo
            met_pos = self.df_notes["metrical_position"].as_matrix()[note_idx]
        #
        #     # iterate over all notes in current solo
        #     for nid in range(num_notes):
        #         met_pos_mat.append(self.decode_metrical_position(met_pos[nid])[1:] + [note_idx[nid], ])
        #
        #     # decode metrical positions
        #     met_pos_mat = np.array(met_pos_mat)
        #     division = met_pos_mat[:, 0]
        #     bar_number = met_pos_mat[:, 1]
        #     beat_number = met_pos_mat[:, 2]
        #     tatum_number = met_pos_mat[:, 3]
        #     note_idx_valid = met_pos_mat[:, 4]
        #
        #     intensity_first_eight = []
        #     intensity_second_eight = []
        #
        #     # iterate over all bars
        #     for bar_id in np.unique(bar_number):
        #
        #         # iterate over all beats within current bar, where eight notes exist
        #         for beat_id in np.unique(beat_number[np.logical_and(bar_number == bar_id,
        #                                                             division == 2)]):
        #
        #             # get note index of current eight-note-pair
        #             note_id_cand = np.where(np.logical_and(bar_number == bar_id,
        #                                                    beat_number == beat_id))[0]
        #             note_idx_cand = note_idx_valid[note_id_cand]
        #
        #             # corresponding tatum values
        #             tatum_cand = tatum_number[note_id_cand]
        #             assert len(tatum_cand) < 3
        #
        #             # check if we have 2 successive eight notes
        #             if 1 in tatum_cand and 2 in tatum_cand:
        #                 # note indices of first and second eight note
        #                 note_idx_first_eight_note = note_idx_cand[np.where(tatum_cand == 1)[0][0]]
        #                 note_idx_second_eight_note = note_idx_cand[np.where(tatum_cand == 2)[0][0]]
        #
        #                 # save corresponding intensity values
        #                 intensity_first_eight.append(intensity_values[note_idx_first_eight_note])
        #                 intensity_second_eight.append(intensity_values[note_idx_second_eight_note])
        #
        #     intensity_first_eight = np.array(intensity_first_eight)
        #     intensity_second_eight = np.array(intensity_second_eight)
        #
        #     # paired t-test (compute difference between groups and run one-sample t test)
        #     t, p = ttest_rel(intensity_first_eight,
        #                      intensity_second_eight)
        #
        #     # cohen's d (effect size measure for paired t-test) -> 0.2 (small), 0.5 (medium), 0.8 (large)
        #     d, mean_diff = cohen_d(intensity_first_eight,
        #                            intensity_second_eight)
        #
        #     # store results of t-test
        #     all_p.append(p)  # significance level
        #     all_d.append(d * np.sign(mean_diff))  # signed effect size
        #     all_n.append(len(intensity_first_eight))  # number of eight-note-pairs in solo
        #
        # all_d = np.array(all_d)
        # all_p = np.array(all_p)
        # all_n = np.array(all_n)
        #
        # # select solos with
        # #  - significant difference between first and second eight notes &
        # #  - minimum of 10 eight-note pairs
        # idx_select = np.where(np.logical_and(all_p < 0.05,
        #                                      all_n > min_num_notes))[0]
        #
        # print("{} solos with positive d, {} solos with negative d, total = {}".format(np.sum(all_d[idx_select] > 0),
        #                                                                               np.sum(all_d[idx_select] < 0),
        #                                                                               len(all_d)))
        #
        # # create table with the N solos of both categories with the highest absolute effect size
        # txt = TextResults()
        # idx_pos = (all_d[idx_select] > 0).nonzero()[0]
        # idx_neg = (all_d[idx_select] < 0).nonzero()[0]
        #
        # idx_pos_and_neg = (idx_pos, idx_neg)
        #
        # # iterate over solos with positive and negative effect size
        # for k, idx_curr_category in enumerate(idx_pos_and_neg):
        #     solo_idx_curr_category = idx_select[idx_curr_category]
        #
        #     # sort solos of current selection (pos. or neg. d values) in descending order based on absolute value
        #     sort_idx = np.argsort(np.abs(all_d[solo_idx_curr_category]))
        #     if k == 0:
        #         sort_idx = sort_idx[::-1]
        #     solo_idx_curr_category = solo_idx_curr_category[sort_idx]
        #
        #     # create row entries in table
        #     for _ in range(num_examples_table):
        #
        #         # avoid overflow
        #         if _ < len(sort_idx):
        #             # write solo metadata into row
        #             idx = solo_idx_curr_category[_]
        #             txt.add("{} & {} & {:2.1f} & {} \\\\".format(self.df_solos['performer'].as_matrix()[idx],
        #                                                          self.df_solos['title'].as_matrix()[idx],
        #                                                          all_d[idx],
        #                                                          generate_p_value_string(all_p[idx])))
        #     if k == 0:
        #         txt.add("\hline")
        #
        # fn_result = os.path.join(self.dir_results_root, 'SISA_Loudness_First_vs_Second_Eight_for_table.txt')
        # txt.save(fn_result)
        # print('{} saved ...'.format(fn_result))

    def intensity_analysis_correlations(self, num_highest_and_lowest_artists=7):

        num_solos = len(self.unique_melids)

        param_labels = ('Pitch', 'Duration', 'RelPosInPhrase')
        num_labels = len(param_labels)
        r = np.zeros((num_labels, num_solos))
        p = np.zeros((num_labels, num_solos))

        phrase_id = self.df_notes['phraseid'].as_matrix()
        intensity = self.df_notes['intensity_solo_max'].as_matrix()
        pitch = self.df_notes['pitch'].as_matrix()
        duration = self.df_notes['duration'].as_matrix()

        # iterate over solos
        for sid, melid in enumerate(self.unique_melids):

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

    def create_performer_data_frame(self, min_num_solos_per_artist=3):
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

        df_performer = df_performer.ix[df_performer['num_solos'] >= min_num_solos_per_artist]

        return df_performer
