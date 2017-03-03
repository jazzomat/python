from scipy.signal import decimate
import numpy as np
import os

from ...transform.transformer import Transformer

__author__ = 'Jakob Abesser'


class F0TrackerPYIN:
    """ Wrapper to call pYin VAMP plugin for f0 contour tracking
    Reference:
        https://code.soundsoftware.ac.uk/projects/pyin
    """

    def __init__(self):
        pass
    #
    # def _down_sampling(self):
    #     """ Signal downsampling
    #     """
    #     if self.down_sample_fac != 1.:
    #         self.samples = decimate(self.samples, int(1/self.down_sample_fac))
    #         self.sample_rate *= self.down_sample_fac
    #
    # def _define_time_axis(self):
    #     """ Define global time axis, use frame centers for time stamps
    #     """
    #     self.time_axis_samples = np.arange(self.num_frames) * self.hopsize + .5*self.blocksize
    #     self.time_axis_sec = self.time_axis_samples / float(self.sample_rate)

    def process(self,
                fn_wav,
                pitch,
                onset,
                duration,
                tuning_frequency=440.):
        """ Perform fundamental frequency tracking based on given score information (note parameters)
        :param fn_wav: (string) WAV file name
        :param pitch: (list / ndarray) Note MIDI pitch values
        :param onset: (list / ndarray) Note onset times in secnods
        :param duration:  (list / ndarray) Note durations in secnods
        :param tuning_frequency: (float) Tuning frequency (e.g. previously estimated using the pymus package)
        :return: global_freq_hz: (ndarray) Frame-wise f0 values in Hz (0 where no contour is present)
        :return: time_axis_sec: (ndarray) Corresponding time stamps in seconds
        :return: contours (dict) Dict with contours parameters (size: num_contours) with following keys:
            'f0_hz': (ndarray) frame-wise f0 values in Hz
            'f0_cent_rel': (ndarray) frame-wise f0 deviations from annotated pitch in cent
            't_sec': (ndarray) Time-stamps in sec
        """
        # enforce numpy array format
        if type(pitch) is not np.ndarray:
            pitch = np.array(pitch)
        if type(onset) is not np.ndarray:
            onset = np.array(onset)
        if type(pitch) is not np.ndarray:
            duration = np.array(duration)

        # generate sonic-annotator command and excecute it
        fn_csv = os.path.join(os.path.dirname(fn_wav),
                              'pyin.csv')
        # add sonic annotator to PATH if necessary
        # os.system('export "PATH=/Applications/SonicAnnotator/:$PATH"')
        command = "sonic-annotator -d vamp:pyin:pyin:smoothedpitchtrack \"{}\" -w csv --csv-one-file {} --csv-force --csv-separator ';'".format(fn_wav,
                                                                                                                                                fn_csv)
        os.system(command)

        # read result (tuning frequency)
        raw_data = np.loadtxt(fn_csv, delimiter=';', usecols=(1, 2), comments='@') # comment parameter needed if # is in filename

        time_axis_sec = raw_data[:, 0]
        f0_hz = raw_data[:, 1]

        num_notes = len(onset)
        contours = [None for _ in range(num_notes)]
        for n in range(num_notes):
            f0 = tuning_frequency*2**((pitch[n]-69)/12.)
            curr_contour = dict()
            mask = np.where(np.logical_and(time_axis_sec >= onset[n],
                                           time_axis_sec <= onset[n] + duration[n]))[0]
            if np.sum(mask) > 0:
                curr_contour['f0_hz'] = f0_hz[mask]
                curr_contour['f0_cent_rel'] = 1200*np.log2(f0_hz[mask] / f0)
                curr_contour['t_sec'] = time_axis_sec[mask]
            else:
                curr_contour['f0_hz'] = None
                curr_contour['f0_cent_rel'] = None
                curr_contour['t_sec'] = None
            contours[n] = curr_contour

        if os.path.isfile(fn_csv):
            os.remove(fn_csv)

        return f0_hz, time_axis_sec, contours

