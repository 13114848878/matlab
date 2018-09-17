clear;clc
filepath = 'C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\data';
name = 'c.bdf';
EEG = pop_readbdf( [filepath,'\',name]);
data = EEG.data;

fs = 1000;
nFB = 1;
data = data(1:8,1:10*fs);
data_filted = bp40_FB1(double(data)',fs,nFB);


% EEG = pop_eegfiltnew(EEG,1,45);
% data_filted = EEG.data;
% data_filted = data_filted';

window = hanning(fs);
noverlap = [];
nfft = fs;
range = 'onesided';
data_len = size(data,2);
w_l = 1*fs;
%n_i = data_len-w_l
n = 0;
for i = 1:w_l*0.25:data_len-w_l
    n = n + 1;
    p = pwelch(data_filted(i:i+w_l,:),window,noverlap,nfft,fs,range);
    p = p';
    psd(:,:,n) = p(:,1:41);
end
c_psd_10s = log(psd);
plot(c_psd_10s(:,:,15)')

save c_psd_10s c_psd_10s

