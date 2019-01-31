% BCI Competition data cleaner
% Clean the data and save it as matlab matlab mat file. 
% 
% record of revisions :
%      date           programmer          description of change
%      ====           ==========          =====================
%    01/02/2019    Mehrdad Kashefi            Original Program 
% ...................................................................
% define variables:
%.............................................
%cleaning up.
clc;
clear;
close all;
%% Loading Data
load('~/Datasets/BCI_IV_2b/BCI_IV_2b.mat');
for sub =4:4
    subject_count = sub;
    window_size = 64;
    timelap = 14; % Stride
    overlap = window_size - timelap;
    fs = 250;
    num_channel = 3;
    do_normalize = 1; % wheater or not perform normalization

    FreqMu = linspace(6,13,16);  %16 freq sample for Mu band
    FreqBeta = linspace(17,30,15); % 15 freq samples for Beta band

    % Fix this later
    data_processed = zeros((2*120)+160,3*31,32);  % Session* Tials , channel* Freq , time_winodw
    label_processed = [];
    % data_processed = [];
    trial_count = 1;
    for Sess= 1:3 
        Data = Subject(subject_count).Session(Sess).Data;
        Label = Subject(subject_count).Session(Sess).Label;

        for tr=1:size(Data,1)    % Loop over 120 trials
            freq_slice_clannel = [];
            for ch=1:num_channel % Loop over 3 channels
                data_slice = Data(tr,250:end-250,ch);
                ISNAN = sum(isnan(data_slice));
                ISZERO = sum(data_slice) == 0;
                [s_mu,f_mu,t_mu] = spectrogram(data_slice,window_size,overlap,FreqMu,fs);
                [s_beta,f_beta,t_beta] = spectrogram(data_slice,window_size,overlap,FreqBeta,fs);
                if do_normalize
                    s_beta_nomalized = s_beta - min(s_beta(:));
                    s_beta_nomalized = s_beta_nomalized ./ max(s_beta_nomalized(:));
                    s_beta = s_beta_nomalized;

                    s_mu_nomalized = s_mu - min(s_mu(:));
                    s_mu_nomalized = s_mu_nomalized ./ max(s_mu_nomalized(:));
                    s_mu = s_mu_nomalized;

                end
                freq_slice = [s_beta;s_mu];
                freq_slice_clannel= [freq_slice_clannel;freq_slice];
            end  
            % Detecting all zero trials and NaN containing trials
            if ISNAN || ISZERO
                disp(['Is there NaN in Data ',num2str(ISNAN)])
                disp(['Is all data zeros ',num2str(ISZERO)])
            else
                data_processed(trial_count,:,:) = abs(freq_slice_clannel);
                trial_count = trial_count +1;
                label_processed = [label_processed; Label(tr)];
            end
    %         imagesc(abs(freq_slice_clannel))
    %         drawnow
    %         pause(0.1)
    %         disp(trial_count)
        end
    end
    disp(["The final number of the trials is ",trial_count-1]);
    % Removing the extra values in data_processed (This extra value exists due
    % to some zero trials)
    data_processed = data_processed(1:trial_count-1,:,:);
    % Saving the data
    save(['data_subjet_', num2str(subject_count)],'data_processed','label_processed')
end
class_1 = data_processed(label_processed == 0,  :,:);
class_2 = data_processed(label_processed == 1,  :,:);
class_1_mean = squeeze(mean(class_1,1));
class_2_mean = squeeze(mean(class_2,1));
imagesc(class_1_mean)
figure
imagesc(class_2_mean)