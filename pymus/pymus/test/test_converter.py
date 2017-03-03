import unittest
from ..convert.converter import Converter

__author__ = 'Jakob Abesser'


class TestConvert(unittest.TestCase):
    """ Unit tests for Convert class
    """

    def setUp(self):
        pass

    def test_pitch_to_octave(self):
        """ Unit test for pitch_to_octave() """
        self.assertEqual(Converter.pitch_to_octave(0), -1)
        self.assertEqual(Converter.pitch_to_octave(11), -1)
        self.assertEqual(Converter.pitch_to_octave(12), 0)
        self.assertEqual(Converter.pitch_to_octave(23), 0)
        self.assertEqual(Converter.pitch_to_octave(24), 1)
        self.assertEqual(Converter.pitch_to_octave(48), 3)

    def test_pitch_to_chroma(self):
        """ Unit test for pitch_to_chroma() """
        self.assertEqual(Converter.pitch_to_chroma(0), 0)
        self.assertEqual(Converter.pitch_to_chroma(3), 3)
        self.assertEqual(Converter.pitch_to_chroma(11), 11)
        self.assertEqual(Converter.pitch_to_chroma(12), 0)
        self.assertEqual(Converter.pitch_to_chroma(25), 1)

    def test_midi_pitch_to_note_name(self):
        """ Unit test for midiPitch2NoteName() """
        self.assertEqual(Converter.pitch_to_note_name(24), 'c1')
        self.assertEqual(Converter.pitch_to_note_name(24, delimiter=' '), 'c 1')
        self.assertEqual(Converter.pitch_to_note_name(24, delimiter='-'), 'c-1')
        self.assertEqual(Converter.pitch_to_note_name(25, upper_case=True), 'Db1')
        self.assertEqual(Converter.pitch_to_note_name(25, upper_case=False), 'db1')
        self.assertEqual(Converter.pitch_to_note_name(25, accidental='#'), 'c#1')
        self.assertEqual(Converter.pitch_to_note_name(25, accidental='b'), 'db1')

    def test_midi_pitch_to_freq(self):
        """ Unit test for midiPitch2Freq() """
        self.assertEqual(Converter.pitch_to_freq(69), 440.)

    def test_freq_to_midi_pitch(self):
        """ Unit test for freq2MidiPitch() """
        self.assertEqual(Converter.freq_to_pitch(440.), 69)


if __name__ == "__main__":
    unittest.main()
