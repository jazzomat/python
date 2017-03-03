import numpy as np

from .loudness_critical_band_approximation import LoudnessCriticalBandApproximationIsolatedMonophonicTracks
from ...tools import Tools


__author__ = 'Jakob Abesser'


class LoudnessEstimator:
    """ Wrapper class for loudness estimators
    """

    @staticmethod
    def process(samples,
                sample_rate,
                pitch,
                onset,
                duration,
                **options):
        """ Call note-wise loudness estimation
        :param samples: (ndarray) Audio samples
        :param sample_rate: (float) Sampling rate
        :param pitch: (ndarray) MIDI pitch values
        :param onset: (ndarray) Note onset values in seconds
        :param duration: (ndarray) Note duration values in seconds
        :param options: (dict)
        :returns results: (dict) note-wise loudness results with keys:
            'max': peak loudness over note duration
            'meadian': median loudness over note duration
            'std': standard deviation over note duration
            'tempCentroid': temporal centroid of loudness over note duration [0,1]
            'relPeakPos': relative peak position over note duration [0,1]
        """
        if type(pitch) is not np.ndarray:
            pitch = np.array(pitch)
        if type(onset) is not np.ndarray:
            onset = np.array(onset)
        if type(pitch) is not np.ndarray:
            duration = np.array(duration)

        if options['loudness_estimation_method'] == 'critical_band_approximation':
            loudness, time_sec = LoudnessCriticalBandApproximationIsolatedMonophonicTracks.process(samples,
                                                                                                   sample_rate,
                                                                                                   **options)
        else:
            raise Exception("Loudness estimation method not implemented!")

        # aggregate loudness values over note durations
        return Tools.aggregate_framewise_function_over_notes(loudness, time_sec, onset, duration)
