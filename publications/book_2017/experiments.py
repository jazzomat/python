import pandas as pd
import os
import numpy as np
import time
import seaborn as sns
import pandas as pn
import matplotlib.pyplot as pl
sns.set(style="white", color_codes=True)
from scipy import stats

from tools import AudioAnalysisTools, TextWriter

__author__ = 'Jakob Abesser'


class AudioAnalysisExperiments:

    def __init__(self,
                 dir_data,
                 dir_results):
        self.dir_results = dir_results
        self.dir_data = dir_data

        # data loading & preparation
        self.df_notes, self.df_beats, self.df_solos = self.load_data()
        self.df_performer = self.create_performer_data_frame()

        self.tools = AudioAnalysisTools
        self.extractors = {'metadata': [self.metadata_analysis],
                           'tuning': [self.tuning_analysis_tuning_freq_vs_recording_year],
                           'intensity': [self.intensity_analysis_correlations]}
        self.text_writer = TextWriter()

    def load_data(self):
        return [pd.read_pickle(os.path.join(self.dir_data, 'df_%s.dat' % _)) for _ in ['notes', 'beats', 'solos']]

    def run(self):
        for category in self.extractors.keys():
            for extractor in self.extractors[category]:
                extractor()

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
                        str(np.sum(self.df_performer['num_notes'])) + "} \\")
        self.text_writer.save(os.path.join(self.dir_results,
                                      'metadata_analysis_artist_stats.txt'))

    def tuning_analysis_tuning_freq_vs_recording_year(self):
        delta_f_ref_440_cent = np.abs(1200.*np.log2(self.df_solos['tuning_frequency']/440.))
        year = self.df_solos['recording_year']

        # linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(year, delta_f_ref_440_cent)

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

    def intensity_analysis_correlations(self):

        num_solos = self.df_solos.shape[0]

        # param_labels = ('Pitch', 'Duration', 'RelPosInPhrase')
        # num_labels = len(param_labels)
        # r = np.zeros((num_labels, num_solos))
        # p = np.zeros((num_labels, num_solos))
        #
        # phrase_id_values = self.df_notes['phraseid'].as_matrix()
        # intensity_values = self.df_notes['intensity_max'].as_matrix()
        # pitch_values = self.df_notes['pitch'].as_matrix()
        # duration_values = self.df_notes['duration'].as_matrix()
        #
        # base_name_values = self.df_notes['base_name'].as_matrix()
        # intensity_within_percentiles_values = self.df_notes['intensity_within_percentiles'].as_matrix()
        #
        # # iterate over solos
        # for s in range(num_solos):
        #
        #     print('Solo {} / {}'.format(s + 1, num_solos))
        #
        #     # get note indices of current solo with valid intensity values
        #     base_name = self.df_solos['base_name'].as_matrix()[s]
        #     note_idx = np.where(self.df_notes['base_name'] == base_name)[0]
        #
        #     # get note parameters
        #     curr_intensity = intensity_values[note_idx]
        #     curr_duration = duration_values[note_idx]
        #     curr_pitch = pitch_values[note_idx]
        #     curr_phrase_id = phrase_id_values[note_idx]
        #     assert all(np.diff(curr_phrase_id) >= 0)
        #     curr_rel_pos_in_phrase = get_relative_position_in_phrase(curr_phrase_id)
        #
        #     vec = (curr_pitch, curr_duration, curr_rel_pos_in_phrase)
        #     N = len(vec)
        #     # compute Pearson correlation coefficient
        #     for _ in range(N):
        #         # TODO check for normal distribution
        #         r[_, s], p[_, s] = pearsonr(curr_intensity, vec[_])
        #
        # # check for significant correlation
        # sig_mask = p < .05
        #
        # # average r over solos for each artist with significant correlations
        # artist = np.array(self.df_solos['performer'].as_matrix())
        # unique_artist = np.unique(artist)
        # num_unique_artist = len(unique_artist)
        #
        # r_artist_mean = np.zeros((num_labels, num_unique_artist))
        # r_artist_std = np.zeros((num_labels, num_unique_artist))
        # r_artist_valid = np.ones((num_labels, num_unique_artist), dtype=bool)
        #
        # for aid, a in enumerate(unique_artist):
        #     solo_idx = np.where(artist == a)[0]
        #
        #     for lid in range(num_labels):
        #         # focus on significant correlations
        #         solo_idx = solo_idx[p[lid, solo_idx] < .05]
        #
        #         if len(solo_idx) > 0:
        #
        #             r_artist_mean[lid, aid] = np.mean(r[lid, solo_idx])
        #             r_artist_std[lid, aid] = np.std(r[lid, solo_idx])
        #         else:
        #             r_artist_valid[lid, aid] = False
        #             print('NOPE for ' + a + " - " + str(lid))
        #
        # # TODO remove: hack such that non-valid entries get mean values so they don't show up in the table
        # for aid in range(num_unique_artist):
        #     for lid in range(num_labels):
        #         if not r_artist_valid[lid, aid]:
        #             r_artist_mean[lid, aid] = np.mean(r_artist_mean[lid, :])
        #
        # # get matrix with row-wise solo idx in descending order of correlation coefficient size
        # sorted_indices = np.zeros((num_labels, num_unique_artist))
        # for lid in range(num_labels):
        #     sorted_indices[lid] = np.argsort(r_artist_mean[lid, :])[::-1]
        #
        # num_highest_and_lowest_artists = 7
        # txt_results = TextResults()
        # ranks = np.concatenate((np.arange(num_highest_and_lowest_artists),
        #                         np.arange(num_unique_artist - num_highest_and_lowest_artists, num_unique_artist)))
        #
        # for _ in ranks:
        #
        #     txt_results.add(
        #         "{} & {} \\newline ({:1.2f}~$\pm$~{:1.2f}) & {} \\newline ({:1.2f}~$\pm$~{:1.2f}) & {} \\newline ({:1.2f}~$\pm$~{:1.2f}) \\\\".format(
        #             _ + 1,
        #             unique_artist[sorted_indices[0, _]],
        #             r_artist_mean[0, sorted_indices[0, _]],
        #             r_artist_std[0, sorted_indices[0, _]],
        #             unique_artist[sorted_indices[1, _]],
        #             r_artist_mean[1, sorted_indices[1, _]],
        #             r_artist_std[1, sorted_indices[1, _]],
        #             unique_artist[sorted_indices[2, _]],
        #             r_artist_mean[2, sorted_indices[2, _]],
        #             r_artist_std[2, sorted_indices[2, _]]))
        #     if _ == num_highest_and_lowest_artists - 1:
        #         txt_results.add('\hline')
        # fn_result = os.path.join(self.dir_results_root, 'SISA_loudness_vs_performers_select.txt')
        # txt_results.save(fn_result)
        # print('{} saved ...'.format(fn_result))

    def create_performer_data_frame(self):
        unique_performer = np.unique(self.df_solos['performer'])
        num_unique_performer = len(unique_performer)
        unique_performer_inst = ['' for _ in unique_performer]
        unique_performer_num_solos = np.zeros(num_unique_performer, dtype=int)
        unique_performer_num_notes = np.zeros(num_unique_performer, dtype=int)
        # get instrument(s), number of solos, number of notes for each unique performer
        for id, up in enumerate(unique_performer):
            unique_performer_inst[id] = list(set(self.df_solos['instrument'][self.df_solos['performer'] == up]))
            unique_performer_num_solos[id] = np.sum(self.df_solos['performer'] == up)
            melids = self.df_solos['melid'][self.df_solos['performer'] == up]
            for melid in melids:
                unique_performer_num_notes[id] += len(self.df_notes['onset'][self.df_notes['melid'] == melid])

        # collect results in pandas dataframe
        df_performer = pd.DataFrame({'instrument': unique_performer_inst,
                                     'num_notes': unique_performer_num_notes,
                                     'num_solos': unique_performer_num_solos},
                                    index=unique_performer)

        return df_performer
