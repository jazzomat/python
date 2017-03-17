import os
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
from tools import AnalysisTools, TextWriter
from sklearn.ensemble import ExtraTreesClassifier
import scipy.stats

__author__ = 'Jakob Abesser'


class SymbolicAnalysisExperiments:

    def __init__(self,
                 dir_data,
                 dir_results,
                 fontsize=14,
                 num_features_to_select=15):
        self.dir_results = dir_results
        self.dir_data = dir_data

        # data loading & preparation
        self.raw_data, \
        self.metadata_feature_labels, \
        self.metadata_features, \
        self.numeric_feature_labels, \
        self.numeric_features = self.load_data()

        self.extractors = {'feature_selection': [self.one_vs_n_feature_selection]}
        self.fontsize = fontsize
        self.num_features_to_select = num_features_to_select
        self.text_writer = TextWriter()
        self.tools = AnalysisTools

    def load_data(self):
        """ Load features exported with MeloSpySuite GUI and split it into metadata and numeric features """
        fn_csv = os.path.join(self.dir_data, 'classification_features_chorus_level.csv')
        raw_data = pd.read_csv(fn_csv, sep=';')

        # split features into metadata features and numeric features
        metadata_feature_labels = ['full_title',
                                   'id',
                                   'instrument',
                                   'performer',
                                   'rhythmfeel',
                                   'seg_id',
                                   'seg_type',
                                   'style',
                                   'title',
                                   'tonality_type']
        metadata_features = raw_data.filter(metadata_feature_labels, axis=1).as_matrix()

        numeric_feature_labels = [_ for _ in list(raw_data.columns) if _ not in metadata_feature_labels]
        numeric_features = raw_data.filter(numeric_feature_labels, axis=1).as_matrix()

        return raw_data, metadata_feature_labels, metadata_features, numeric_feature_labels, numeric_features

    def run(self):
        for category in self.extractors.keys():
            for extractor in self.extractors[category]:
                extractor()
        print('Finished all experiments :)')

    def one_vs_n_feature_selection(self):
        """ Perform 1-vs-N feature selection to identify most characteristic properties
            of individual classes """
        # feature check
        assert np.all(np.logical_not(np.isnan(self.numeric_features)))

        num_items, num_features = self.numeric_features.shape

        clf = ExtraTreesClassifier()

        # attributes of interest
        class_type_labels = ['instrument', 'performer', 'rhythmfeel', 'style', 'tonality_type']

        for ctid, class_type in enumerate(class_type_labels):

            print('Run 1-vs-N feature selection experiment for class type = %s' % class_type)

            self.text_writer.reset()

            # prepare class id
            metadata_feat_idx = self.metadata_feature_labels.index(class_type)
            all_feature_values = self.metadata_features[:, metadata_feat_idx]
            class_id, unique_class_values = self.create_class_ids(all_feature_values)
            num_classes = len(unique_class_values)

            for cid, class_label in enumerate(unique_class_values):

                # define 1-vs-N class IDs
                class_id_curr = np.ones(num_items, dtype=int)
                class_id_curr[class_id == cid] = 0

                self.text_writer.add("%s; (N = %d vs. %d)" % (class_label,
                                                             int(np.sum(class_id_curr == 0)),
                                                             int(np.sum(class_id_curr == 1))))

                self.text_writer.add("Rank; Feature; Mean (class); Mean (others); Significance (t-test); Cohen's D")

                clf.fit(self.numeric_features, class_id_curr)

                # indices of best features
                best_feat_idx = np.argsort(clf.feature_importances_)[::-1][:self.num_features_to_select]

                # t-test for significance
                for fid, numeric_feat_idx in enumerate(best_feat_idx):
    
                    # compute t-Test to compare if means differ significantly
                    t, p = scipy.stats.ttest_ind(self.numeric_features[class_id_curr == 0, numeric_feat_idx],
                                                 self.numeric_features[class_id_curr == 1, numeric_feat_idx])
    
                    # only consider features with significant differences in means
                    if p < 0.05:
                        self.text_writer.add('%d; %s; %f; %f; %s; %f' % (fid,
                                                                         self.numeric_feature_labels[numeric_feat_idx],
                                                                         float(np.mean(self.numeric_features[class_id_curr == 0, numeric_feat_idx])),
                                                                         float(np.mean(self.numeric_features[class_id_curr == 1, numeric_feat_idx])),
                                                                         self.tools.generate_p_value_string(p),
                                                                         self.tools.cohens_d(self.numeric_features[class_id_curr == 0, numeric_feat_idx],
                                                                                             self.numeric_features[class_id_curr == 1, numeric_feat_idx])))
            self.text_writer.save(os.path.join(self.dir_results, 'symbolic_analysis_1_vs_N_feature_selection_%s.csv' % class_type))

    def create_class_ids(self, all_vals):
        """ Generate class ID vector from given set of item-wise values.
        Args:
            all_vals (ndarray): Item-wise annotations (num_items)
        Returns:
            unique_vals (ndarray): Unique annotations (num_classes)
            class_id (ndarray): Item-wise class ids (num_items)
        """
        class_id = np.zeros(len(all_vals), dtype=int)
        unique_vals = np.unique(all_vals)
        for i, val in enumerate(unique_vals):
            class_id[all_vals == val] = i
        return class_id, unique_vals