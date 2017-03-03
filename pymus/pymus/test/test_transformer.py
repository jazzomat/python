__author__ = 'Jakob Abesser'

import unittest
import numpy as np
import os

from ..transform.transformer import Transformer
from ..tools import Tools


class TestTransform(unittest.TestCase):
    """ Unit tests for Transformer class
    """

    def setUp(self):
        """ Load reference data exported from Matlab """
        self.data = dict()
        refData = dict()
        refData['test_stft'] = ['x', 'spec_real', 'spec_imag', 'blocksize', 'hopsize', 'NFFT']
        refData['test_reass_spec'] = ['x', 'spec', 'f', 'fs', 'if', 'blocksize', 'hopsize', 'NFFT']

        for label in refData:
            self.data[label] = dict()
            for param in refData[label]:
                self.data[label][param] = np.loadtxt(Tools.get_file_path_for_test_data(label + '_' + param + '.txt'),
                                                     delimiter=',',
                                                     dtype=float)

    def test_sftf(self):
        """ Unit test for Transformer.stft() """
        spec, timeFrameSec, freqBinHz = Transformer.stft(self.data['test_stft']['x'],
                                                         int(self.data['test_stft']['blocksize']),
                                                         int(self.data['test_stft']['hopsize']),
                                                         int(self.data['test_stft']['NFFT']))

        # check dimensions
        np.testing.assert_almost_equal(spec.real, self.data['test_stft']['spec_real'], decimal=12)
        np.testing.assert_almost_equal(spec.imag, self.data['test_stft']['spec_imag'], decimal=12)

        # check for correct values
        self.assertEqual(spec.shape[0], len(freqBinHz))
        self.assertEqual(spec.shape[1], len(timeFrameSec))

    def test_reassigned_spec(self):
        """ Unit test for Transformer.reassigned_spec() """
        spec, freqBinHz, timeFrameSec, instFreqHz = Transformer.reassigned_spec(self.data['test_reass_spec']['x'],
                                                                                self.data['test_reass_spec']['blocksize'],
                                                                                self.data['test_reass_spec']['hopsize'],
                                                                                sample_rate=self.data['test_reass_spec']['fs'],
                                                                                freq_bin_hz=self.data['test_reass_spec']['f'],
                                                                                n_fft=self.data['test_reass_spec']['NFFT'])

        # check for correct values
        np.testing.assert_equal(freqBinHz, self.data['test_reass_spec']['f'])
        np.testing.assert_almost_equal(spec, self.data['test_reass_spec']['spec'], decimal=12)
        np.testing.assert_almost_equal(instFreqHz, self.data['test_reass_spec']['if'], decimal=5)

        # check for correct values
        self.assertEqual(spec.shape[0], len(freqBinHz))


if __name__ == "__main__":
    unittest.main()
