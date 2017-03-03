function create_test_data_stft

% create_test_data_stft
% Generate data for stft unit test in pymus python package
% Author: Jakob Abesser
% E-Mail: 3vs5@posteo.de

close all

fnWAV = 'test_transform.wav';

% load audio file
[x,fs] = wavread(fnWAV);
x = mean(x,2);

blocksize = 1024;
hopsize = 512;

% lets use some zero-padding
NFFT = 2048; 

% compute STFT (use hann to get Hanning window with first and last sample
% being zero for easy implementation)
X = spectrogram(x, hann(blocksize), blocksize-hopsize, NFFT);
       
% save variables to be imported in Python unit test
dlmwrite('test_stft_x.txt', x, 'delimiter', ',', 'precision', 16);
dlmwrite('test_stft_spec_real.txt', real(X), 'delimiter', ',', 'precision', 16);
dlmwrite('test_stft_spec_imag.txt', imag(X), 'delimiter', ',', 'precision', 16);
dlmwrite('test_stft_hopsize.txt', hopsize, 'delimiter', ',', 'precision', 16);
dlmwrite('test_stft_blocksize.txt', blocksize, 'delimiter', ',', 'precision', 16);
dlmwrite('test_stft_NFFT.txt', NFFT, 'delimiter', ',', 'precision', 16);

disp('done :)')