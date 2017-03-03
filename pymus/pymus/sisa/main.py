import numpy as np

from .f0_tracking.estimator import F0Tracker
from .tuning.estimator import TuningEstimator
from .loudness.estimator import LoudnessEstimator
from ..tools import Tools

__author__ = 'Jakob Abesser'

"""
Score-informed Solo Analysis - SISA
"""


class ScoreInformedSoloAnalysis:

    """ Class allows for score-informed analysis of melodies based on given score / transcription
        information
    """

    def __init__(self):
        pass

    def analyze(self,
                fn_wav_tuning_estimation,
                fn_wav_f0_tracking,
                fn_wav_intensity_estimation,
                pitch,
                onset,
                duration,
                do_estimate_tuning_frequency=False,
                do_estimate_f0_contours=False,
                do_estimate_loudness=False,
                **options):
        """ Run score-informed solo analysis
        :param fn_wav_tuning_estimation: (string) Audio file as input to tuning estimation
        :param fn_wav_f0_tracking: (string) Audio file as input to f0 tracking
        :param fn_wav_intensity_estimation: (string) Audio file as input to intensity estimation
        :param pitch: (ndarray): Note MIDI pitch values
        :param onset: (ndarray): Note onset times (s)
        :param duration: (ndarray): Note durations (s)
        :param do_estimate_tuning_frequency: (bool) Switch to perform tuning frequency estimation
        :param do_estimate_f0_contours: (bool) Switch to perform f0 tracking
        :param do_estimate_loudness: (bool) Switch to perform loudness estimation
        """

        if not (do_estimate_f0_contours or do_estimate_loudness or do_estimate_tuning_frequency):
            print("No processing mode activated!")
            return None

        # spectral estimation steps
        results = dict()

        # tuning estimation
        if do_estimate_tuning_frequency:

            options = Tools.set_missing_values(options,
                                               tuning_estimation_method='mauch_nnls',
                                               fn_wav=fn_wav_tuning_estimation)

            # estimate tuning frequency
            results['tuning_frequency'] = TuningEstimator.process(**options)
        else:
            if 'tuning_frequency' not in results.keys():
                results['tuning_frequency'] = 440.

        # score-informed f0 tracking
        if do_estimate_f0_contours:

            options = Tools.set_missing_values(options,
                                               tuning_frequency=results['tuning_frequency'],
                                               f0_tracking_method='abesser_dafx_2014')

            # load file
            samples, sample_rate = Tools.load_wav(fn_wav_f0_tracking, mono=True)

            results['f0_hz'], \
            results['f0_time_sec'], \
            results['f0_contours'] = F0Tracker.process(samples,
                                                       sample_rate,
                                                       pitch,
                                                       onset,
                                                       duration,
                                                       **options)


        # score-informed loudness estimation
        if do_estimate_loudness:

            options = Tools.set_missing_values(options,
                                               loudness_estimation_method='critical_band_approximation')

            # load file
            samples, sample_rate = Tools.load_wav(fn_wav_intensity_estimation, mono=True)

            results['loudness'] = LoudnessEstimator.process(samples,
                                                            sample_rate,
                                                            pitch,
                                                            onset,
                                                            duration,
                                                            **options)

        return results


