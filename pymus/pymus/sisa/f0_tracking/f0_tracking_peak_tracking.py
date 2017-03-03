from scipy.signal import decimate
import numpy as np

from ...transform.transformer import Transformer

__author__ = 'Jakob Abesser'


class F0TrackerPeakTrackingAbesserDAFX2014:
    """ Implementation of score-informed fundamental frequency tracking proposed in
    Reference:
        J. Abesser, M. Pfleiderer, K. Frieler, W.-G. Zaddach: "SCORE-INFORMED TRACKING AND CONTEXTUAL
        ANALYSIS OF FUNDAMENTAL FREQUENCY CONTOURS IN TRUMPET AND SAXOPHONE JAZZ SOLOS", Proc. of the
        17th Int. Conference on Digital Audio Effects (DAFx-14), Erlangen, Germany, September 1-5, 2014
    """

    def __init__(self,
                 blocksize=2048,
                 hopsize=128,
                 zero_padding_factor=8,
                 bins_per_octave=25*12,
                 down_sampling_factor=.5,
                 lower_mag_th_db=50.,
                 pitch_margin=2,
                 verbose=True,
                 visualize=False):
        """ Class initialization
        :param blocksize: (int) Blocksize in samples
        :param hopsize: (int) Hopsize in samples
        :param zero_padding_factor: (int) Zero-padding factor for STFT
        :param bins_per_octave: (int) Frequency resolution in bins / octave for log-frequency axis
        :param down_sampling_factor: (int) Downsampling factor
        :param lower_mag_th_db: (float) Lower magnitude boundary in dB
        :param pitch_margin: (int) Pitch margin in semitones around target pitch to investigate in the tracking
        :param verbose: (bool) Switch for verbose output
        :param visualize: (bool) Switch for visualization
        """
        self.blocksize = float(blocksize)
        self.hopsize = float(hopsize)
        self.hopsize_correction = int(self.blocksize/self.hopsize)-1
        self.zero_padding_factor = zero_padding_factor
        self.bins_per_octave = bins_per_octave
        self.down_sample_fac = down_sampling_factor
        self.lower_mag_th_db = lower_mag_th_db
        self.pitch_margin = pitch_margin
        self.verbose = verbose
        self.visualize = visualize

        # maximum allowed absolute deviation between neighbored frames
        self.delta_bin_max = np.round(.2*self.bins_per_octave/12.)

        # initialize
        self.samples = None
        self.sample_rate = None
        self.tuning_frequency = None
        self.num_frames = 0
        self.num_samples = 0
        self.num_notes = 0
        self.global_f0_hz = None

    def _down_sampling(self):
        """ Signal downsampling
        """
        if self.down_sample_fac != 1.:
            self.samples = decimate(self.samples, int(1/self.down_sample_fac))
            self.sample_rate *= self.down_sample_fac

    def _define_time_axis(self):
        """ Define global time axis, use frame centers for time stamps
        """
        self.time_axis_samples = np.arange(self.num_frames) * self.hopsize + .5*self.blocksize
        self.time_axis_sec = self.time_axis_samples / float(self.sample_rate)

    def process(self,
                samples,
                sample_rate,
                pitch,
                onset,
                duration,
                tuning_frequency=440.):
        """ Perform fundamental frequency tracking based on given score information (note parameters)
        :param samples: (ndarray) Monaural sample vector
        :param sample_rate: (int / float) Sample rate in Hz
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
        if type(duration) is not np.ndarray:
            duration = np.array(duration)

        self.num_notes = len(pitch)
        self.samples = samples
        self.sample_rate = sample_rate
        self.tuning_frequency = tuning_frequency

        # note offsets
        offset = np.array([onset[_] + duration[_] for _ in range(self.num_notes)])

        # downsampling
        self._down_sampling()

        self.num_samples = len(self.samples)
        self.num_frames = self.num_samples / self.hopsize

        # set global time axis
        self._define_time_axis()
        self.global_f0_hz = np.zeros(len(self.time_axis_sec))

        # iterate over all notes
        contours = []
        for n in range(self.num_notes):

            if self.verbose:
                if n % 20 == 1:
                    print('Process note {}/{} ...'.format(n, self.num_notes))

            # track f0 contour for current note
            f0_hz, f0_cent_rel, time_frame_idx = self._track_f0_contour(onset[n],
                                                                        offset[n],
                                                                        pitch[n])

            # save f0 contour
            curr_contour = dict()
            curr_contour['f0_hz'] = f0_hz
            curr_contour['f0_cent_rel'] = f0_cent_rel
            curr_contour['t_sec'] = self.time_axis_sec[time_frame_idx]
            contours.append(curr_contour)

            # save contour values to global contour
            self.global_f0_hz[time_frame_idx] = f0_hz

        return self.global_f0_hz, self.time_axis_sec, contours

    def _note_boundaries_in_global_time_axis(self, onset, offset):
        """ Find note boundaries based on onset and offset time w.r.t. global time axis
        :param onset: (float) Onset time in secnods
        :param offset: (float) Offset time in seconds
        :return: onset_frame_idx: (int) Note start frame index
        :return: offset_frame_idx: (int) Note stop frame index
        """
        start_sample = np.max([0, np.round(onset*self.sample_rate) - .5*self.blocksize])
        end_sample = np.min([self.num_samples-1, np.round(offset*self.sample_rate) + .5*self.blocksize])
        return int(round(start_sample/self.hopsize)), \
               int(round(end_sample/self.hopsize))

    def _scale_and_limit_spectrogram(self, spec):
        """ Scale and limit magnitude spectrogram
        :param spec: (2d ndarray) Magnitude spectrogram
        :return: spec_scaled: (2d ndarray) Scaled magnitude spectrogram
        """
        # scale spectrogram to [0,1]
        spec -= np.min(spec)
        spec /= np.max(spec)

        # scale to dB magnitude
        spec = 20*np.log10(spec + np.spacing(1))

        # limit to lower threshold
        spec[spec < -self.lower_mag_th_db] = -self.lower_mag_th_db

        return spec

    def _track_f0_contour(self,
                          onset,
                          offset,
                          pitch):
        """
        Score-informed note-wise f0 tracking using the reassigned spectrogram
        :param onset: (float) Note onset (sec)
        :param offset: (float) Note offsec (sec)
        :param pitch: (int) Note MIDI pitch
        :return f0_hz: (ndarray) Frame-wise f0 values
        :return f0_cent_rel: (ndarray) Frame-wise deviations of f0 from target pitch (given
                                       by note MIDI pitch and tuning frequency) in cent
        :return frames: (ndarray) Frame indices (of global time axis, see self.time_axis_samples)
        """
        start_frame, end_frame = self._note_boundaries_in_global_time_axis(onset, offset)

        # define logarithmically spaced frequency axis
        log_f_axis_midi = np.arange(pitch-self.pitch_margin,
                                    pitch+self.pitch_margin+12./self.bins_per_octave,
                                    12./self.bins_per_octave)
        log_f_axis_hz = self.tuning_frequency*2**((log_f_axis_midi-69.)/12.)

        # compute reassigned spectrogram
        start_sample = self.time_axis_samples[start_frame]
        end_sample = self.time_axis_samples[end_frame]
        spec, _, time_frame_sec, inst_freq_hz = Transformer.reassigned_spec(self.samples[start_sample:end_sample],
                                                                            self.blocksize,
                                                                            self.hopsize,
                                                                            sample_rate=self.sample_rate,
                                                                            freq_bin_hz=log_f_axis_hz,
                                                                            n_fft=self.zero_padding_factor*self.blocksize)

        # scale & limit spectrogram
        spec = self._scale_and_limit_spectrogram(spec)

        # fundamental frequency according to pitch & tuning frequency
        f0_target_hz = self.tuning_frequency*2**((pitch-69)/12.)

        # get absolute distance to f0 for all frame-wise magnitude maxima
        bin_max = np.argmax(spec, axis=0)
        abs_dist_to_f0_bin = np.abs(f0_target_hz - log_f_axis_hz[bin_max])

        # find optimal starting frame for forwards-backwards tracking (look
        # for frame with maximum peak closest to the annotated f0 value)
        # (don't use argmin here since there could be multiple bins of interest)
        start_frame_tracking = (abs_dist_to_f0_bin == np.min(abs_dist_to_f0_bin)).nonzero()[0]

        # if we have multiple potential start positions -> take peak with highest magnitude
        if len(start_frame_tracking) > 1:
            mag = spec[bin_max[start_frame_tracking], start_frame_tracking]
            start_frame_tracking = start_frame_tracking[np.argmax(mag)]

        # perform forwards-backwards tracking based on continuity assumption
        f0_bin = self._forwards_backwards_tracking(spec, start_frame_tracking, bin_max[start_frame_tracking], self.delta_bin_max)

        # frame-wise fundamental frequency values
        f0_hz = log_f_axis_hz[f0_bin]

        # frame-wise f0 deviation from annotated pitch in cent
        f0_cent_rel = (f0_bin - self.pitch_margin*self.bins_per_octave/12.)*1200./self.bins_per_octave

        # plot tracking results over spectrogram
        if self.visualize:
            import matplotlib.pyplot as plt
            plt.matshow(spec)
            plt.hold(True)
            plt.plot(bin_max, 'r-')
            plt.show(block=False)

        return f0_hz, f0_cent_rel, np.arange(start_frame, start_frame + len(f0_hz))

    def _forwards_backwards_tracking(self, spec, idx_start, bin_start, delta_bin_max):
        """ Forwards-backwards tracking of peak contour in a given magnitude spectrogram using a start position and a
            proximity criterion
        :param spec: (2d ndarray) Magnitude spectrogram (num_bins x num_frames)
        :param idx_start: (int) Start frame index
        :param bin_start: (int) Start frequency bin
        :param delta_bin_max: (int) (onse-sided) search margin from one to the next frame
        :return: peak_bin: (ndarray) Peak bin positions (num_frames)
        """
        num_bins, num_frames = spec.shape
        peak_bin = np.zeros(num_frames)
        peak_bin[idx_start] = bin_start

        # backwards tracking
        if idx_start > 0:
            for f in range(idx_start-1, -1, -1):
                peak_bin[f] = self._find_proximate_peak(spec[:, f],
                                                        peak_bin[f+1],
                                                        delta_bin_max)

        # forwards tracking
        if idx_start < num_frames - 1:
            for f in range(idx_start+1, num_frames):
                peak_bin[f] = self._find_proximate_peak(spec[:, f],
                                                        peak_bin[f-1],
                                                        delta_bin_max)

        return peak_bin.astype(int)

    def _find_proximate_peak(self, spec, peak_bin_ref, delta_bin_max):
        """ Find peak in spectogram frame based on proximity towards peak bin from previous frame (peak tracking using
            proximity criterion)
        :param spec: (ndarray) Magnitude spectrogram frame
        :param peak_bin_ref: (int) Reference peak bin from previous frame
        :param delta_bin_max: (int) (one-sided) search margin around peak_bin_ref
        :return: peak_bin (int) Peak bin position in current frame
        """
        min_bin, max_bin = self._get_margin_bin_boundaries(peak_bin_ref,
                                                           delta_bin_max,
                                                           len(spec))
        return min_bin + np.argmax(spec[min_bin:max_bin+1])

    @staticmethod
    def _get_margin_bin_boundaries(mid_bin, bin_margin, num_bins):
        """ Find lower and upper bin boundaries based on given middle bin (mid_bin), a bin margin (bin_margin) and the
            axis boundaries (0, max_bin)
        :param mid_bin: (int) Middle bin
        :param bin_margin: (int) (one-sided) bin margin
        :param num_bins: (int) Number of bins in axis
        :return: min_bin: (int) Lower boundary bin index
        :return: max_bin: (int) Upper boundary bin index
        """
        return int(np.max((0, mid_bin-bin_margin))), int(np.min((num_bins-1, mid_bin+bin_margin)))
