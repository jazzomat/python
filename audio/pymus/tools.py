# try to import libsoundfile for audio import, alternatively try loading audio via scipy or wave
try:
    import soundfile
    soundfile_found = True
except ImportError:
    from scipy.io.wavfile import read
    import wave
    soundfile_found = False

import numpy as np
import os

__author__ = 'Jakob Abesser'


class Tools:
    """ Class provides several tools for audio analysis
    """

    def __init__(self):
        pass

    @staticmethod
    def set_missing_values(options,
                           **default_frame_wise_values):
        """ Add default frame_wise_values & keys to dictionary if frame_wise_values are not set
        Args:
            options (dict): Arbitrary dictionary (e.g. containing processing options)
            default_frame_wise_values (dict): Keyword list with default frame_wise_values to be set if corresponding
                                              keys are not set in options dict
        Returns:
            options (dict): Arbitrary dictionary with added default frame_wise_values if required
        """
        for param in default_frame_wise_values.keys():
            if param not in options:
                options[param] = default_frame_wise_values[param]
        return options

    @staticmethod
    def load_wav(fn_wav,
                 mono=False):
        """ Function loads samples from WAV file. Both implementations (wave / scipy package) fail for some WAV files 
            hence we combine them.
        Args:
            fn_wav (string): WAV file name
            mono (bool): Switch if samples shall be converted to mono
        Returns:
            samples (np array): Audio samples (between [-1,1]
                                 > if stereo: (2D ndarray with DIM numSamples x numChannels),
                                 > if mono: (1D ndarray with DIM numSamples)
            sample_rate (float): Sampling frequency [Hz]
        """
        if soundfile_found:
            samples, sample_rate = soundfile.read(fn_wav)
        else:
            try:
                samples, sample_rate = Tools._load_wav_file_via_scipy(fn_wav)
            except:
                try:
                    samples, sample_rate = Tools._load_wav_file_via_wave(fn_wav)
                except:
                    raise Exception("WAV file could neither be opened using Scipy nor Wave!")

        # mono conversion
        if mono:
            if samples.ndim == 2:
                if samples.shape[1] > 1:
                    samples = np.mean(samples, axis=1)
                else:
                    samples = np.squeeze(samples)

        # scaling
        if np.max(np.abs(samples)) > 1:
            samples = samples.astype(float) / 32768.0

        return samples, sample_rate

    @staticmethod
    def _load_wav_file_via_wave(fn_wav):
        """ Load samples & sample rate from WAV file """
        fp = wave.open(fn_wav)
        num_channels = fp.getnchannels()
        num_frames = fp.getnframes()
        frame_string = fp.readframes(num_frames*num_channels)
        data = np.fromstring(frame_string, np.int16)
        samples = np.reshape(data, (-1, num_channels))

        sample_rate = float(fp.getframerate())
        return samples, sample_rate

    @staticmethod
    def _load_wav_file_via_scipy(fn_wav):
        """ Load samples & sample rate from WAV file """
        inputData = read(fn_wav)
        samples = inputData[1]
        sample_rate = inputData[0]
        return samples, sample_rate

    @staticmethod
    def aggregate_framewise_function_over_notes(frame_wise_values,
                                                time_sec,
                                                onset,
                                                duration):
        """ Aggregate a frame-wise function (e.g. loudness) over note durations to obtain note-wise features
        :param frame_wise_values: (ndarray) Frame-wise values
        :param time_sec: (ndarray) Time frame frame_wise_values in seconds
        :param onset: (ndarray) Note onset times in seconds
        :param duration: (ndarray) Note durations in seconds
        :return: result: (dict of ndarrays) Note-wise aggregation results with keys
                    'max': Maximum over note duration
                    'median': Median over note duration
                    'std': Standard deviation over note duration
                    'temp_centroid': Temporal centroid over note duration [0,1]
                    'rel_peak_pos': Position of global maximum over note duration relative to note duration [0,1]
        """
        dt = time_sec[1]-time_sec[0]
        num_notes = len(onset)

        onset_frame = (onset/dt).astype(int)
        offset_frame = ((onset+duration)/dt).astype(int)

        # initialize
        result = dict()
        result['max'] = np.zeros(num_notes)
        result['median'] = np.zeros(num_notes)
        result['std'] = np.zeros(num_notes)
        result['temp_centroid'] = np.zeros(num_notes)
        result['rel_peak_pos'] = np.zeros(num_notes)

        for n in range(num_notes):
            curr_frame_wise_values = frame_wise_values[onset_frame[n]:offset_frame[n]+1]
            num_frames_curr = len(curr_frame_wise_values)
            result['max'][n] = np.max(curr_frame_wise_values)
            result['median'][n] = np.median(curr_frame_wise_values)
            result['std'][n] = np.std(curr_frame_wise_values, ddof=1)  # same result as in Matlab
            result['temp_centroid'][n] = np.sum(np.linspace(0, 1, num_frames_curr)*curr_frame_wise_values) /\
                                         np.sum(curr_frame_wise_values)
            if num_frames_curr > 1:
                result['rel_peak_pos'][n] = float(np.argmax(curr_frame_wise_values))/(num_frames_curr-1)
            else:
                result['rel_peak_pos'][n] = 0

        return result

    @staticmethod
    def quadratic_interpolation(x):
        """ Peak refinement using quadratic interpolation.
        Args:
            x (ndarray): 3-element numpy array. Central element was identified as local peak before 
                         (e.g. using numpy.argmax)
        Returns:
            peak_pos (float): Interpolated peak position (relative to central element)
            peak_val (float): Interpolated peak value
        """
        peak_pos = (x[2] - x[0])/(2*(2*x[1] - x[2] - x[0]))
        peak_val = x[1] - 0.25*(x[0]-x[2])*peak_pos
        return peak_pos, peak_val

    @staticmethod
    def moving_average_filter(x,
                              N=5):
        """ Moving average on a vector
        Args:
            x (ndarray): Input vector
            N (int): Filter length
        Returns
            y (ndarry): Filtered vector
        """
        return np.convolve(x, np.ones((N,))/N, mode='valid')

    @staticmethod
    def acf(x):
        """ Autocorrelation function
        Args:
            x (ndarray): Input function
        Returns:
            acf (ndarray): Autocorrelation function
        """
        result = np.correlate(x, x, mode='full')
        return result[result.size/2:]

    @staticmethod
    def get_file_path_for_test_data(fn):
        """ Get path for test data
        :param fn: file name
        :return: absolute path of filename in subfolder /test/data
        """
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test', 'data', fn)
