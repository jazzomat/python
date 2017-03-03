import unittest
import numpy as np

from ..sisa.main import ScoreInformedSoloAnalysis
from ..tools import Tools

__author__ = 'Jakob Abesser'


class TestScoreInformedSoloAnalysis(unittest.TestCase):

    def setUp(self):
        """ Load reference data from txt files exported from Matlab via dlmwrite
        """
        self.sisa = ScoreInformedSoloAnalysis()

        self.fn_wav = Tools.get_file_path_for_test_data('test_sisa.wav')

        # load reference transcription
        self.data = dict()
        for param in ['duration', 'onset', 'pitch', 'loudness', 'loudness_median',
                      'loudness_rel_peak_pos', 'loudness_std', 'loudness_temp_centroid',
                      'tuning_freq']:
            self.data[param] = np.loadtxt(Tools.get_file_path_for_test_data(param + '.txt'),
                                          delimiter=',',
                                          dtype=float)

    def test_load_wav(self):
        """ Unit test for audio import: check samples & sample rate
        """
        samples, sample_rate = Tools.load_wav(Tools.get_file_path_for_test_data('test_transform.wav'), mono=True)
        ref_samples = np.loadtxt(Tools.get_file_path_for_test_data('test_stft_x.txt'))
        np.testing.assert_equal(samples, ref_samples)
        self.assertEqual(sample_rate, 44100)

    # def test_loudness_critical_band_approximation(self):
    #     """ Unit test for critical band approximation method  """
    #     # return
          # TODO: (state 10.12.2015) most values are perfectly equal, however 2-3 deviate
    #     results = ScoreInformedSoloAnalysis().analyze(self.fn_wav,
    #                                                   self.data["pitch"],
    #                                                   self.data["onset"],
    #                                                   self.data["duration"],
    #                                                   do_estimate_tuning_frequency=False,
    #                                                   do_estimate_f0_contours=False,
    #                                                   do_estimate_loudness=True,
    #                                                   loudness_estimation_method='critical_band_approximation')
    #     #
    #     # # test loudness results
    #     test_num_decimals = 5
    #     np.testing.assert_almost_equal(results['loudness']['max'], self.data['loudness'], decimal=test_num_decimals)
    #     np.testing.assert_almost_equal(results['loudness']['median'], self.data['loudness_median'], decimal=test_num_decimals)
    #     np.testing.assert_almost_equal(results['loudness']['std'], self.data['loudness_std'], decimal=test_num_decimals)
    #     np.testing.assert_almost_equal(results['loudness']['temp_centroid'], self.data['loudness_temp_centroid'], decimal=test_num_decimals)
    #     np.testing.assert_almost_equal(results['loudness']['rel_peal_pos'], self.data['loudness_rel_peak_pos'], decimal=test_num_decimals)


if __name__ == '__main__':
    unittest.main()


