pymus - Audio & Music analysis tools
------------------------------------

A Python library including several tools for automatic music analysis.
Special focus is on algorithms for score-informed analysis of melodies in audio recordings of musical instruments.

sisa/
-----

Methods for score-informed analysis

sisa/f0_tracking
----------------

Score-informed tracking of the fundamental frequency contour of each note in a transcribed melody recording.

sisa/loudness
-------------

Score-informed estimation of note-wise intensity values based on a critical band approximation

sisa/tuning
-----------

Wrapper to call NNLS VAMP plugin by Matthias Mauch using sonic-annotator (must be installed)

convert/converter
-----------------

Converter functions between MIDI pitch, frequencies, and note names

features/f0_contour_features
----------------------------

Audio features that characterize (note-wise) fundamental frequency contours. These can be used to train machine
learning models to classify pitch modulation techniques such as bending, slide, vibrato etc.

transform/transformer
---------------------

Implementations of the Short-time Fourier Transform (based on spectrogram function from Matlab) and the Reassigned
Spectrogram using the instantaneous frequency. The latter is useful for frequency tracking since it exhibits sharper
peaks for harmonic signal components compared to the STFT.

wrapper/sonic_visualiser.py
---------------------------

Currently just one function to export time-series to CSV files which can be loaded into Sonic Visualiser for
visualisation purposes (time values layer)

