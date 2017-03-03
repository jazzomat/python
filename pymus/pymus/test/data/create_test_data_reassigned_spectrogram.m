function create_test_data_reassigned_spectrogram

% create_test_data_reassigned_spectrogram
% Generate data for reassignedSpectrogram() unit test in pymus python package
% Author: Jakob Abesser
% E-Mail: 3vs5@posteo.de

close all

fnWAV = 'test_transform.wav';

% load audio file
[x,fs] = wavread(fnWAV);
x = mean(x,2);

% parameters
minMIDI = 20;
maxMIDI = 100;
binsPerOctave = 36;
vFLogMIDI = minMIDI:12/binsPerOctave:maxMIDI;
vFLogHz = 440.*2.^(((vFLogMIDI)-69)./12);
blocksize = 1024;
hopsize = 512;

% lets use some zero-padding
NFFT = 2048; 

% compute IF spectrogram
[mSpecIF,d1,d2,d3,mFReassigned] = reassigned_spectrogram(x, blocksize, hopsize, fs, vFLogHz,...
                                                     'zeroPaddingFactor',2,...
                                                     'methodIF','IF_Abe',...
                                                     'methodAccumulation','round');



% save variables to be imported in Python unit test
dlmwrite('test_reassSpec_x.txt', x, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_spec.txt', mSpecIF, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_if.txt', mFReassigned, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_fs.txt', fs, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_f.txt', vFLogHz, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_hopsize.txt', hopsize, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_blocksize.txt', blocksize, 'delimiter', ',', 'precision', 16);
dlmwrite('test_reassSpec_NFFT.txt', NFFT, 'delimiter', ',', 'precision', 16);

disp('done :)')