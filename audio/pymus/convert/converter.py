from __future__ import division
import math

__author__ = 'Jakob Abesser'


class Converter:
    """ Different conversion methods
    """

    note_names = dict()
    note_names['b'] = ['c', 'db', 'd', 'eb', 'e', 'f', 'gb', 'g', 'ab', 'a', 'bb', 'b']
    note_names['#'] = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

    def __init__(self):
        pass

    @staticmethod
    def pitch_to_note_name(pitch, delimiter='', accidental='b', upper_case=False):
        """ Convert MIDI pitch to note name (note spelling + octave number)
        :param pitch: (int) MIDI pitch value
        :param delimiter: (string) Delimiter between note spelling & octave number
        :param accidental: (string) Key accidental ('b' or '#')
        :param upper_case: (bool) Switch to generate uppercase note names
        :return: (string) Note name
        """
        noteName = "{}{}{}".format(Converter.note_names[accidental][Converter.pitch_to_chroma(pitch)],
                                   delimiter,
                                   Converter.pitch_to_octave(pitch))
        if upper_case:
            noteName = noteName[0].upper() + noteName[1:]
        return noteName

    @staticmethod
    def pitch_to_octave(pitch):
        """ Converts MIDI pitch to octave number
        :param pitch: (int) MIDI pitch
        :return: octave: (int) Octave number
        """
        return int(math.floor(pitch/12)-1)

    @staticmethod
    def pitch_to_chroma(pitch):
        """ Convert MIDI pitch to chroma value
        :param pitch: (int) MIDI pitch
        :return: (int) Chroma value
        """
        return pitch % 12

    @staticmethod
    def pitch_to_freq(pitch):
        """ Convert MIDI pitch to fundamental frequency in Hz
        :param pitch: (int / float) MIDI pitch value
        :return: freq: (float) Fundamental frequency in Hz
        """
        return 440.*2**((pitch-69)/12)

    @staticmethod
    def freq_to_pitch(freq):
        """ Convert MIDI pitch to fundamental frequency in Hz
        :param freq (float): Frequency in Hz
        :return: pitch (float): Corresponding (fractional) MIDI pitch value
        """
        return 69 + 12*math.log(freq/440)/math.log(2)

