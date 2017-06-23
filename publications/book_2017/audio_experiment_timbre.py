import os
import numpy as np
import pickle

__author__ = 'Jakob Abesser'

from symbolic_experiments import SymbolicAnalysisExperiments

if __name__ == '__main__':

    dir_data = '/Users/jakobabeer/Sync/Jakob/Programming/Repositories/2017_jazzomat_python/python/publications/book_2017/data'
    dir_results = '/Users/jakobabeer/Sync/Jakob/Programming/Repositories/2017_jazzomat_python/python/publications/book_2017/results'

    sa = SymbolicAnalysisExperiments(dir_data, dir_results, num_estimators=50, min_effect_size=.3)

    for instrument in ('tp', 'as', 'ts'):

        print('Do feature selection for instrument {}'.format(instrument))

        fn_feat = os.path.join('data', 'timbre_feat_{}.dat'.format(instrument))
        with open(fn_feat, 'rb') as f:
            feat, feat_labels, class_id, class_labels = pickle.load(f)

        # check feature matrix for nans
        assert np.all(np.logical_not(np.isnan(feat)))

        num_classes = len(np.unique(class_id))
        num_items, num_features = feat.shape

        sa.text_writer.reset()
        class_one_label = 'all'

        for cid in range(num_classes):

            # define 1-vs-N class IDs
            class_id_curr = np.ones(num_items, dtype=int)
            class_id_curr[class_id == cid] = 0

            SymbolicAnalysisExperiments.analyze_features_for_two_classes(sa.clf,
                                                                         sa.text_writer,
                                                                         feat,
                                                                         feat_labels,
                                                                         class_id_curr,
                                                                         [class_labels[cid], class_one_label],
                                                                         num_features_to_select=sa.num_features_to_select,
                                                                         min_feature_importance=sa.min_feature_importance,
                                                                         min_effect_size=sa.min_effect_size)

        sa.text_writer.save(os.path.join(dir_results, 'audio_analysis_timbre_instrument_{}_1_vs_N_feature_selection.csv'.format(instrument)))
