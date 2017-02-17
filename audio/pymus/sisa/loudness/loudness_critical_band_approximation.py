from __future__ import division
import numpy as np

from ...tools import Tools
from ...transform.transformer import Transformer

__author__ = 'Jakob Abesser'


class LoudnessCriticalBandApproximationIsolatedMonophonicTracks:

    @staticmethod
    def process(samples,
                sampleRate,
                **options):
        """ Score-informed note loudness estimation from isolated instrument tracks as used in
            J. Abesser, E. Cano, K. Frieler & M. Pfleiderer: Dynamics in Jazz Improvisation -
            Score-informed Estimation and Contextual Analysis of Tone Intensities in Trumpet and Saxophone Solos,
            in Proceedings of the 9th Conference on Interdisciplinary Musicology (CIM), 2014, Berlin, Germany
        :param samples: (ndarray) Audio samples
        :param sampleRate: (float) Sampling rate
        :param options: (dict)
        :return: loudness (ndarray) Frame-wise loudness values
        :return: timeSec (ndarray) Frame values in seconds
        """
        options = Tools.set_missing_values(options,
                                           loudness_critical_band_blocksize=512,
                                           loudness_critical_band_hopsize=480)

        # STFT magnitude spectrogram
        spec, time_frame_sec, freq_bin_hz = Transformer.stft(samples,
                                                             options["loudness_critical_band_blocksize"],
                                                             options["loudness_critical_band_hopsize"],
                                                             use_frame_center_time=True)
        mag_spec = np.absolute(spec)

        # power spectrogram
        power_spec = np.square(np.true_divide(mag_spec, options["loudness_critical_band_blocksize"]))

        num_frames = mag_spec.shape[1]

        # critical band edges in Hz
        band_edges = np.array([1, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700,
                              3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
        num_bands = len(band_edges)-1

        # bin indices of critical bands
        band_bins = np.ceil(band_edges*options["loudness_critical_band_blocksize"]/sampleRate)-1

        # band-wise intensity values
        band_intensities = np.zeros((num_bands, num_frames))
        for b in range(num_bands):
            band_intensities[b, :] = np.sum(power_spec[band_bins[b]:band_bins[b+1]+1, :], axis=0)

        # overall loudness
        loudness = 90.302 + 10*np.log10(np.sum(band_intensities, axis=0)+np.spacing(1))

        return loudness, time_frame_sec
