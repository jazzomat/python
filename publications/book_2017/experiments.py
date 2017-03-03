import pandas as pd
import os
import numpy as np
import time
import seaborn as sns
import pandas as pn
import matplotlib.pyplot as pl
sns.set(style="white", color_codes=True)
from scipy import stats

from tools import AudioAnalysisTools

__author__ = 'Jakob Abesser'


class AudioAnalysisExperiments:

    def __init__(self,
                 dir_data,
                 dir_results):
        self.dir_results = dir_results
        self.dir_data = dir_data
        self.df_notes, self.df_beats, self.df_solos = self.load_data()
        self.tools = AudioAnalysisTools

    def load_data(self):
        return [pd.read_pickle(os.path.join(self.dir_data, 'df_%s.dat' % _)) for _ in ['notes', 'beats', 'solos']]

    def run(self):
        for extractor in [self.tuning_analysis]:
            extractor()

    def tuning_analysis(self):
        for extractor in [self.tuning_analysis_tuning_freq_vs_recording_year]:
            extractor()

    def tuning_analysis_tuning_freq_vs_recording_year(self):
        delta_f_ref_440_cent = np.abs(1200.*np.log2(self.df_solos['tuning_frequency']/440.))
        year = self.df_solos['recording_year']

        # linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(year, delta_f_ref_440_cent)
        self.tools.write_to_text_file(os.path.join(self.dir_results, 'tuning_analysis_tuning_freq_vs_recording_year_lin_reg_r.txt'), "%.3f" % r_value)
        self.tools.write_to_text_file(os.path.join(self.dir_results, 'tuning_analysis_tuning_freq_vs_recording_year_lin_reg_p.txt'), p_value, is_prob=True)


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


