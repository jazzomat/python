function create_sisa_test_data

% create_sisa_test_data
% Generate test data for SISA python package

% Author: Jakob Abesser, nudelsalat@posteo.de
% Created: May 2015

maxDur = 20;
dirOut = '';

RELOAD_FILE_AND_TRANSCRIPTION = 0;

if RELOAD_FILE_AND_TRANSCRIPTION
    [cDataset] = jazzomat_export_dataset_for_WAV_root_folder;

    fnWAV = cDataset{1}.fnWAV;

    [x,fs] = wavread(fnWAV);

    if size(x,2) > 1
        x = mean(x,2);
    end

    x = x(1:round(fs*maxDur));
    fnWAVTarget = fullfile(dirOut,'test_sisa.wav');
    wavwrite(x, fs, fnWAVTarget);

    onset = cDataset{1}.vOnset;
    duration = cDataset{1}.vDuration;
    pitch = cDataset{1}.vPitch;

    offset = onset + duration;
    validIdx = find(offset < maxDur);
    onset = onset(validIdx);
    duration = duration(validIdx);
    pitch = pitch(validIdx);


    dlmwrite(fullfile(dirOut,'onset.txt'), onset, 'delimiter', ',', 'precision', 16)
    dlmwrite(fullfile(dirOut,'duration.txt'), duration, 'delimiter', ',', 'precision', 16)
    dlmwrite(fullfile(dirOut,'pitch.txt'), pitch, 'delimiter', ',', 'precision', 16)
else
   
    onset = dlmread('onset.txt', ',');
    duration = dlmread('duration.txt', ',');
    offset = onset + duration;
    pitch = dlmread('pitch.txt', ',');
    fnWAVTarget = '/Volumes/untitled/Work/Jazzomat_Repository/deployment/sisa/sisa/test/data/test_sisa.wav';
    
end

% loudness estimation
[loudness,...
 loudnessMedian,...
 loudnessRelPeakPos,...
 loudnessStd, ...
 loudnessTemporalCentroid] = get_segment_wise_loudness(fnWAVTarget,...
                                    onset, offset,'normalizeToMaxLoudness',0);
                                
dlmwrite(fullfile(dirOut,'loudness.txt'), loudness, 'delimiter', ',', 'precision', 16)
dlmwrite(fullfile(dirOut,'loudnessMedian.txt'), loudnessMedian, 'delimiter', ',', 'precision', 16)
dlmwrite(fullfile(dirOut,'loudnessRelPeakPos.txt'), loudnessRelPeakPos, 'delimiter', ',', 'precision', 16)
dlmwrite(fullfile(dirOut,'loudnessStd.txt'), loudnessStd, 'delimiter', ',', 'precision', 16)
dlmwrite(fullfile(dirOut,'loudnessTemporalCentroid.txt'), loudnessTemporalCentroid, 'delimiter', ',', 'precision', 16)

% tuning frequency estimation
fA4 = estimate_tuning_frequency(fnWAVTarget);
dlmwrite(fullfile(dirOut,'tuningFreq.txt'), fA4, 'delimiter', ',', 'precision', 16)

% f0 tracking
% cData.fnWAVSolo = fnWAVTarget;
% cData.centerA4Ref = fA4;
% cData.vPitch = pitch;
% cData.vOnset = onset;
% cData.vOffset = onset + duration;
% cContours = jazzomat_score_informed_f0_tracking(cData);
% for k = 1 : length(cContours)
%     dlmwrite(fullfile(dirOut,['f0contour_' num2str(k-1) '_f0Hz.txt']), cContours{k}.vF0Hz, 'delimiter', ',', 'precision', 16)
%    dlmwrite(fullfile(dirOut,['f0contour_' num2str(k-1) '_tSec.txt']), cContours{k}.vTSec, 'delimiter', ',', 'precision', 16)
%    dlmwrite(fullfile(dirOut,['f0contour_' num2str(k-1) '_f0CentRel.txt']), cContours{k}.vF0CentRel, 'delimiter', ',', 'precision', 16)
% end

disp('done :)')
                                