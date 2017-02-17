from __future__ import  division
import unittest
import numpy as np

from ..features.f0_contour_features import ContourFeatures

__author__ = 'Jakob Abesser'


class TestFeatures(unittest.TestCase):
    """ Unit tests for ContourFeatures class
    """

    def setUp(self, show_plot=False):
        """ Generate test vibrato contour
        """
        # hopsize = 512, sample rate = 44.1 kHz
        dt = 512/44100

        # contour length = 100 frames
        N = 100
        n = np.arange(N)

        # fundamental frequency = 440 Hz
        self.f_0_hz = 440

        # modulation frequency = 3 Hz
        self.f_mod_hz = 3

        # modulation lift = 4 Hz
        self.freq_hz = self.f_0_hz + 4.*np.sin(2*np.pi*n*dt*self.f_mod_hz)

        # time frames
        self.time_sec = np.arange(N)*dt

        # frequency deviation from f0 in cent
        self.freq_rel_cent = 1200*np.log2(self.freq_hz/self.f_0_hz)

        if show_plot:
            import matplotlib.pyplot as pl
            pl.figure()
            pl.plot(self.time_sec, self.freq_hz)
            pl.show(block=False)

    def test_contour_features(self):
        """ Unit test for f0_contour_features() """
        features, feature_labels = ContourFeatures().process(self.time_sec,
                                                             self.freq_hz,
                                                             self.freq_rel_cent,
                                                             min_mod_freq_hz=.3)

        assert len(features) == len(feature_labels)

        for _ in range(len(feature_labels)):
            print("{} : {}".format(feature_labels[_], features[_]))

        # test some features
        self.assertAlmostEqual(features[feature_labels.index("mod_freq_hz")], self.f_mod_hz, places=1)
        self.assertGreater(features[feature_labels.index("mod_num_periods")], 5)
        self.assertGreater(features[feature_labels.index("mod_dom")], .5)
        self.assertLess(features[feature_labels.index("lin_f0_slope")], .01)



if __name__ == "__main__":
    unittest.main()
