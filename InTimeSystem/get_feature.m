clear;clc

filepath = 'C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system';
name = 'o_psd_10s';
load([filepath,'\',name,'.mat'])

eval(['psd = ',name,';']);

bw = [1,5,8,13,31];
resolution = 1;
for b_i = 1:length(bw)-1
    bands{b_i} = (bw(b_i)/resolution + 1) : bw(b_i+1)/resolution;
end
n_band = length(bands);
%会自动reshape
AP = squeeze(sum(psd(:,bands{1}(1):bands{end}(end),:),2));
 i = 0;
    for  band = 1:n_band
        band_psd = psd(:,bands{band},:);
        band_sum = squeeze(sum(band_psd,2));
        i = i + 1;
        feature(:,:,i) = band_sum./AP;
    end
size(feature)
feature = permute(feature,[2,1,3]);
shape = size(feature)
o_feature_10s = reshape(feature,shape(1),[]);
save o_feature_10s o_feature_10s


%c_feature_10s_2Hz 代表从2Hz开始计算频带


