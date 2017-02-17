from __future__ import division
import numpy as np

__author__ = 'Jakob Abesser'


class Transformer:
    """ Class implements different time-frequency transformations """

    def __init__(self):
        pass

    @staticmethod
    def stft(samples,
             blocksize,
             hopsize,
             n_fft=None,
             window='Hanning',
             sample_rate=44100,
             use_frame_center_time=False):
        """ Short-time Fourier Transform (STFT), based on re-implementation of Matlab spectrogram() function
        :param samples: (ndarray) Audio samples
        :param blocksize: (int) Blocksize in samples
        :param hopsize: (int) Hopsize in samples
        :param n_fft: (int / None) FFT size in samples (zero-padding is used if necessary)
                     -> if None, n_fft is set to blocksize (no zero-padding)
        :param window: (string) Window type, currently implemented
                        'Hanning': N-point symmetric Hanning window
                        'DiffHanning': Differentiated Hanning window
        :param sample_rate: (float / None) Sampling rate in Hz
                            If None, sample_rate is set to 44100.
        :param use_frame_center_time: (bool) Switch to use frame center as frame time, if False, frame begin is used
        :return: spec: (2d ndarray) Complex STFT spectrogram (nFrequencyBins x nTimeFrames)
        :return: time_frame_sec: (ndarray) Time frame values in seconds
        :return: freq_bin_hz: (ndarray) Frequency bin values in Hz
        """
        if n_fft is None:
            n_fft = blocksize

        # buffer signal
        buffer_mat = Transformer.buffer_signal(samples,
                                               blocksize,
                                               hopsize)
        num_frames = buffer_mat.shape[1]

        # apply windowing
        window = np.reshape(Transformer._window_function(blocksize,
                                                         window), (-1, 1))
        buffer_mat *= window

        # time frames
        if use_frame_center_time:
            time_frame_sec = (np.arange(num_frames, dtype=float)+.5)
        else:
            time_frame_sec = (np.arange(num_frames, dtype=float))
        time_frame_sec *= hopsize/sample_rate

        # frequency bins
        freq_bin_hz = np.arange(n_fft/2+1)*sample_rate/n_fft

        # STFT
        spec = np.fft.fft(buffer_mat, n=int(n_fft), axis=0)[:int(n_fft/2)+1, :]

        # compute FFTs
        return spec, time_frame_sec, freq_bin_hz

    @staticmethod
    def buffer_signal(samples,
                      blocksize,
                      hopsize):
        """ Buffer signal into overlapping or non-overlapping frames. Missing samples are filled with zeros
        :param samples: (ndarray) Sample vector
        :param blocksize: (int) Blocksize in samples
        :param hopsize: (int) Hopsize in samples
        :return: buffer: (2d ndarray) Sample buffer, dimensions: blocksize x num_frames
        """
        overlap = blocksize - hopsize
        num_samples = len(samples)
        num_frames = np.floor((num_samples-overlap)/(blocksize-overlap))

        # create sample buffer matrix with (non-)overlapping frames
        col_idx = np.arange(num_frames) * hopsize
        row_idx = np.reshape(np.arange(blocksize),
                             (-1, 1))
        index = col_idx + row_idx
        index = index.astype(int)

        return samples[index]

    @staticmethod
    def reassigned_spec(samples,
                        blocksize,
                        hopsize,
                        sample_rate=44100,
                        freq_bin_hz=None,
                        n_fft=None):
        """ Compute reassigned magnitude spectrogram by mapping STFT magnitude values to time-frequency
            bins that correspond to the local instantaneous frequency. For harmonic signals, this results
            in a spectrogram representation with a higher sparsity, i.e., sharper harmonic peaks compared
            to the STFT. This is useful for tasks such as pitch tracking & music transcription.
        :param samples: (ndarray) Audio samples
        :param blocksize: (int) Blocksize in samples
        :param hopsize: (int) Hopsize in samples
        :param sample_rate: (int) Sampling rate in Hz
        :param freq_bin_hz: (None / ndarray) Desired frequency axis (bin values in Hz) for reassigned spectrogram,
                          must be linearly-spaced or logarithmically-spaced.
                          This allows for example to define a logarithmically-spaced frequency axis which is
                          used to map the magnitudes to.
                          If None, the common linearly-spaced FFT frequency axis based on the given STFT parameters
                          is used
        :param n_fft: (int / None) FFT size in samples (zero-padding is used if necessary)
                     -> if None, n_fft is set to blocksize (no zero-padding)
        :return: spec: (2d ndarray) Reassigned magnitude spectrogram (nFrequencyBins x nTimeFrames)
        :return: freq_bin_hz: (ndarray) Frequency bin values in Hz of reassigned spectrogram (nFrequencyBins)
        :return: time_frame_sec: (ndarray) Time frame values in seconds (nTimeFrames)
        :return: inst_freq_hz: (2d ndarray) Instantaneous frequency values [Hz]  (nFrequencyBins x nTimeFrames)
        """
        # Comptute magnitude STFT spectrogram & reassigned frequency positions based on the local instantaneous
        # frequency values
        inst_freq_hz, spec_stft, time_frame_sec, freq_bin_hz_stft = Transformer._inst_freq_abe(samples,
                                                                                               blocksize,
                                                                                               hopsize,
                                                                                               sample_rate,
                                                                                               n_fft=n_fft)

        # magnitude spectrogram
        spec_stft = np.abs(spec_stft)

        # use STFT frequency axis as target frequency axis if not specified otherwise
        if freq_bin_hz is None:
            freq_bin_hz = freq_bin_hz_stft
        assert all(np.diff(freq_bin_hz) > 0)

        nBins = len(freq_bin_hz)
        num_frames = spec_stft.shape[1]
        spec = np.zeros((nBins, num_frames))
        dt = time_frame_sec[1] - time_frame_sec[0]

        # only consider instantaneous frequencies within range of target frequency axis
        min_freq_hz = freq_bin_hz[0]
        max_freq_hz = freq_bin_hz[-1]
        idx_bin_valid = np.logical_and(inst_freq_hz >= min_freq_hz, inst_freq_hz <= max_freq_hz).nonzero()
        freq_hz_reassigned = inst_freq_hz[idx_bin_valid[0], idx_bin_valid[1]]
        time_sec_reassigned = time_frame_sec[idx_bin_valid[1]]
        num_valid_entries = len(freq_hz_reassigned)

        freq_scale_spacing_type = Transformer._scale_spacing_type(freq_bin_hz)

        if freq_scale_spacing_type == 'linear':
            df = freq_bin_hz[1] - freq_bin_hz[0]
            freq_bin_frac_reassigned = (freq_hz_reassigned - min_freq_hz)/df
        elif freq_scale_spacing_type == 'logarithmic':
            bins_per_octave = np.round(1./np.log2(freq_bin_hz[2]/freq_bin_hz[1]))
            freq_bin_frac_reassigned = bins_per_octave*np.log2(freq_hz_reassigned/min_freq_hz)
        else:
            raise Exception('Target frequency axis spacing must be linear or logarithmic!')

        # no time reassignment
        time_frame_frac_reassigned = np.round(time_sec_reassigned/dt)

        # simple frequency reassignment by mapping magnitude to target frequency bin which is closest to instantaneous
        # frequency value
        # (alternative would be a fractional mapping of magnitudes to both adjacent bins, this is more time consuming
        # and does not showed improvements in past experiments)
        freq_bin_frac_reassigned = np.round(freq_bin_frac_reassigned)

        for n in range(num_valid_entries):
            spec[freq_bin_frac_reassigned[n], time_frame_frac_reassigned[n]] += spec_stft[idx_bin_valid[0][n], idx_bin_valid[1][n]]

        return spec, freq_bin_hz, time_frame_sec, inst_freq_hz

    @staticmethod
    def _scale_spacing_type(scale):
        """
        :param scale: (ndarray) Scale values (e.g. frequency scale values in Hz)
        :return: type: (string) Scale spacing type ('linear', 'logarithmic', or 'other')
        """
        assert len(scale) > 2
        tol = 1e-10
        if all(np.diff(scale, 2)) < tol:
            return 'linear'
        elif all(np.diff(np.log2(scale), 2) < tol):
            return 'logarithmic'
        return 'other'

    @staticmethod
    def _inst_freq_abe(samples,
                       blocksize,
                       hopsize,
                       sample_rate,
                       n_fft=None):
        """ Compute instantaneous frequency values based on the method proposed in
                Toshihiro Abe et al. in ICASSP'95, Eurospeech'97
        :param samples: (ndarray) Audio samples
        :param blocksize: (int) Blocksize in samples
        :param hopsize: (int) Hopsize in samples
        :param sample_rate: (int) Sampling rate in Hz
        :param n_fft: (int / None) FFT size in samples (zero-padding is used if necessary)
                     -> if None, n_fft is set to blocksize (no zero-padding)
        :return: inst_freq_hz: (2d ndarray) Instantaneous frequency values [Hz]  (nFrequencyBins x nTimeFrames)
        :return: spec: (2d ndarray) STFT spectrogram (nFrequencyBins x nTimeFrames)
                       This can be used for magnitude reassignment as done in reassigned_spec()
        :return: time_frame_sec: (ndarray) Time frame values in seconds (nTimeFrames)
        :return: freq_bin_hz: (ndarray) Frequency bin values in Hz (nFrequencyBins)
        """
        # compute 2 STFTs with Hanning and DiffHanning window
        spec, time_frame_sec, freq_bin_hz = Transformer.stft(samples,
                                                             blocksize,
                                                             hopsize,
                                                             n_fft,
                                                             'Hanning',
                                                             sample_rate=sample_rate)
        spec_diff, _, _ = Transformer.stft(samples,
                                           blocksize,
                                           hopsize,
                                           n_fft,
                                           'DiffHanning',
                                           sample_rate=sample_rate)

        # compute instantaneous frequencies, use array broadcasting, ignore N/2 + 1 frame (fs/2)
        inst_freq_hz = np.imag(spec_diff[:-1, :]/spec[:-1, :])/(2*np.pi) + np.reshape(np.arange(n_fft/2)*sample_rate/n_fft,
                                                                                      (-1, 1))
        return inst_freq_hz, spec, time_frame_sec, freq_bin_hz

    @staticmethod
    def _window_function(N, window, sample_rate=44100.):
        """ Create window functions
        :param N: (int) window width
        :param window: (string) Window type, currently implemented
                'Hanning': N-point symmetric Hanning window with first and last sample being 0 (Matlab: hann())
                'DiffHanning': Differentiated Hanning window
        :param sample_rate: (float) Sampling rate in Hz
        :return: window: (ndarray) window function
        """
        if window == 'Hanning':
            return .5*(1-np.cos(2*np.pi*np.arange(N)/(N-1)))
        elif window == 'DiffHanning':
            return -np.pi*sample_rate / (N-1) * np.sin(2*np.pi*np.arange(N)/(N-1))
        else:
            raise Exception('Non-valid value for window')
