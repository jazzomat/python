from __future__ import division
import numpy as np
from ..tools import Tools

__author__ = 'Jakob Abesser'


class ContourFeatures:
    """ Class implements several fundamental frequency contour features that can for instance be used to
        classify different pitch modulation techniques (vibrato, bend, slide, etc.)
    """

    def process(self,
                time_sec,
                f0_hz,
                f0_cent_rel,
                feat_mod_range_cent=True,
                feat_av_f0_dev=True,
                feat_abs_av_f0_dev=True,
                feat_iqr_cent_rel=True,
                feat_lin_slope_cent_rel=True,
                feat_tendency_hz=True,
                feat_modulation_hz=True,
                min_mod_freq_hz=3,
                max_mod_freq_hz=10,
                mov_av_filter_len=None):
        """ Run feature extraction
        :param time_sec: (ndarray) Time frames (sec)
        :param f0_hz: (ndarray) Frame-wise fundamental frequency values (Hz)
        :param f0_cent_rel: (ndarray) Frame-wise deviation from target pitch (cent)
        :param feat_mod_range_cent: (bool) Switch to compute that feature
        :param feat_av_f0_dev: (bool) Switch to compute that feature
        :param feat_abs_av_f0_dev: (bool) Switch to compute that feature
        :param feat_iqr_cent_rel: (bool) Switch to compute that feature
        :param feat_lin_slope_cent_rel: (bool) Switch to compute that feature
        :param feat_tendency_hz: (bool) Switch to compute that feature
        :param feat_modulation_hz: (bool) Switch to compute that feature
        :param min_mod_freq_hz: (float) Minimum f0 modulation frequency
        :param max_mod_freq_hz: (float) Maximum f0 modulation frequency
        :param mov_av_filter_len: (int / None) Moving average filter length (None if not to be applied)
        :return features: (ndarray) Feature values
        :return feature_labels: (list of strings) Feature dimension labels
        """

        # feature extraction
        features = []
        feature_labels = []
        val_range = np.arange(len(f0_cent_rel))

        if feat_mod_range_cent:
            features.append(np.max(f0_cent_rel)-np.min(f0_cent_rel))
            feature_labels.append('mod_range_cent')

        if feat_av_f0_dev:
            features.append(np.median(f0_cent_rel))
            feature_labels.append('av_f0_dev_median')
            features.append(np.mean(f0_cent_rel))
            feature_labels.append('av_f0_dev_mean')

        if feat_abs_av_f0_dev:
            features.append(np.median(np.abs(f0_cent_rel)))
            feature_labels.append('av_abs_f0_dev_median')
            features.append(np.mean(np.abs(f0_cent_rel)))
            feature_labels.append('av_abs_f0_dev_mean')

        if feat_iqr_cent_rel:
            features.append(np.subtract(*np.percentile(f0_cent_rel, [75, 25])))
            feature_labels.append('iqr_f0_dev')

        if feat_lin_slope_cent_rel:
            features.append(np.polyfit(val_range, f0_cent_rel, 1)[0])
            feature_labels.append('lin_f0_slope')

        if feat_tendency_hz:
            curr_features, curr_labels = ContourFeatures.compute_tendency_features(f0_hz)
            features += curr_features
            feature_labels += curr_labels

        if feat_modulation_hz:
            curr_features, curr_labels = ContourFeatures.compute_modulation_features(time_sec,
                                                                                     f0_hz,
                                                                                     max_mod_freq_hz=max_mod_freq_hz,
                                                                                     min_mod_freq_hz=min_mod_freq_hz,
                                                                                     mov_av_filter_len=mov_av_filter_len)
            features += curr_features
            feature_labels += curr_labels

        return features, feature_labels

    @staticmethod
    def compute_tendency_features(freq_hz):
        """ Compute tendency features over fundamental frequency contour of a given note
        :param freq_hz: (ndarray) Frame-wise fundamental frequency values (Hz)
        :return features: (ndarray) Feature values
        :return feature_labels: (list of strings) Feature dimension labels
        """

        features = []
        feature_labels = []

        if len(freq_hz) < 3:
            num_feat = 7
            return [None for _ in range(num_feat)], ['' for _ in range(num_feat)]

        # Median frequency gradient
        f_diff = np.diff(freq_hz)
        N = float(len(f_diff))
        idx_mid = int(round(N/2.))
        features.append(np.median(f_diff))
        feature_labels.append('median_freq_gradient_overal')
        features.append(np.median(f_diff[:idx_mid]))
        feature_labels.append('median_freq_gradient_first_half')
        features.append(np.median(f_diff[idx_mid:]))
        feature_labels.append('median_freq_gradient_second_half')

        # Segmentation into ascending and descending segments
        diff_sign = np.sign(f_diff)
        num_diff_sign = len(diff_sign)
        seg_sign = []
        seg_len = []
        c = 0
        curr_diff_sign = 1
        for _ in range(num_diff_sign):
            if _ == 0:
                curr_diff_sign = diff_sign[_]
                seg_sign.append(curr_diff_sign)
                c += 1
            else:
                if diff_sign[_] != curr_diff_sign or _ == num_diff_sign-1:
                    seg_len.append(c)
                    c = 0
                    if _ < num_diff_sign - 1:
                        curr_diff_sign = diff_sign[_]
                        seg_sign.append(curr_diff_sign)
                        c = 1
                else:
                    c += 1

        seg_len = np.array(seg_len)
        seg_sign = np.array(seg_sign)

        features.append(float(np.sum(seg_len[np.nonzero(seg_sign == -1.)[0]]))/num_diff_sign)
        feature_labels.append('ratio_of_descending_segments')

        features.append(float(np.sum(seg_len[np.nonzero(seg_sign == 0.)[0]]))/num_diff_sign)
        feature_labels.append('ratio_of_constant_segments')

        try:
            features.append(float(np.sum(seg_len[np.nonzero(seg_sign == 1.)[0]]))/num_diff_sign)
        except:
            features.append(None)
        feature_labels.append('ratio_of_ascending_segments')

        features.append(np.median(seg_len) / np.sum(seg_len))
        feature_labels.append('av_rel_segment_len')

        return features, feature_labels

    @staticmethod
    def compute_modulation_features(time_sec,
                                    freq_hz,
                                    max_mod_freq_hz=10,
                                    min_mod_freq_hz=0.3,
                                    mov_av_filter_len=5):

        features = []
        feature_labels = []

        # moving average filtering
        if mov_av_filter_len is not None:
            freq_hz = Tools.moving_average_filter(freq_hz, N=mov_av_filter_len)

        # remove DC part
        freq_hz -= np.mean(freq_hz)

        # modulation frequency
        mod_freq, mod_dom = ContourFeatures.estimate_modulation_frequency_using_acf(time_sec,
                                                                                    freq_hz,
                                                                                    max_mod_freq_hz=max_mod_freq_hz,
                                                                                    min_mod_freq_hz=min_mod_freq_hz)
        features.append(mod_freq)
        feature_labels.append('mod_freq_hz')

        # modulation dominance measure
        features.append(mod_dom)
        feature_labels.append('mod_dom')

        # number of modulation periods
        features.append(np.sum(np.abs(np.diff(np.diff(freq_hz) > 0))))
        feature_labels.append('mod_num_periods')

        return features, feature_labels

    @staticmethod
    def estimate_modulation_frequency_using_acf(time_sec,
                                                freq_hz,
                                                max_mod_freq_hz=10.,
                                                min_mod_freq_hz=3.):
        """ Compute modulation frequency of given frequency contour using autocorrelation function (ACF)
        :param time_sec: (ndarray) Time frame values in seconds
        :param freq_hz: (ndarray) Frequency contour values in Hz
        :param max_mod_freq_hz: (float) Maximum f0 modulation frequency
        :param min_mod_freq_hz: (float) Minimum f0 modulation frequency
        :return mod_freq: (float) Modulation frequency in Hz
        :return mod_dom: (float) Modulation dominance measure
        """
        dt = time_sec[1] - time_sec[0]
        L = len(freq_hz)

        # auto-correlation function
        acf = Tools.acf(freq_hz)

        acf -= np.min(acf)
        acf /= np.max(acf)

        # ACF lag values in seconds
        lag_sec = np.arange(L)*dt

        # focus on search range
        focus_bins = np.where(np.logical_and(lag_sec > 1./max_mod_freq_hz,
                                             lag_sec < 1./min_mod_freq_hz))[0]
        if len(focus_bins) == 0:
            return None, None

        focus_acf = acf[focus_bins]
        focus_lag_sec = lag_sec[focus_bins]

        # global maximum in search val_range
        max_idx = np.argmax(focus_acf)

        # refine peak position using quadratic interpolation
        if len(focus_bins) > 2 and 0 < max_idx < len(focus_acf) - 1:
            max_idx_int, peak_val = max_idx + Tools.quadratic_interpolation(focus_acf[max_idx-1:max_idx+2])

            # get modulation frequency using linear interpolation
            frac = max_idx_int % 1.
            low_idx = int(np.floor(max_idx_int))
            mod_freq = 1./(focus_lag_sec[low_idx]*frac + focus_lag_sec[low_idx+1]*(1-frac))

            # dominance measure for modulation
            mod_dom = peak_val
        else:
            max_bin = focus_bins[max_idx]
            mod_freq = 1./lag_sec[max_bin]
            mod_dom = acf[max_bin]

        return mod_freq, mod_dom
