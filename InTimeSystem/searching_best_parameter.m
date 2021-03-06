function ACC_all = searching_best_parameter()
filepath = 'C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\data';
name = 'c.bdf';
EEG = pop_readbdf( [filepath,'\',name]);
close_data = EEG.data;
name = 'o.bdf';
EEG = pop_readbdf( [filepath,'\',name]);
open_data = EEG.data;
fs = 1000;
sample_len = 1*fs;
% n = 0;
ACC_all = [];
for i = 1:60
    parfor j = 1:10
        train_data_len = 5*fs*i;
        sample_len = 100*j;
        %     n = n + 1;
        
        [ACC] = preform(close_data,open_data,train_data_len,sample_len);
        ACC_all(i,j) = ACC;
    end
end



function [ACC] = preform(close_data,open_data,train_data_len,sample_len)
[c_tr_psd,c_te_psd] = getPSD(close_data,train_data_len,sample_len);

[o_tr_psd,o_te_psd] = getPSD(open_data,train_data_len,sample_len);

o_tr_f = getFeature(o_tr_psd);
c_tr_f = getFeature(c_tr_psd);
o_te_f = getFeature(o_te_psd);
c_te_f = getFeature(c_te_psd);
[X_train,Y_train]=getLabel(c_tr_f,o_tr_f);
[X_test,Y_test]=getLabel(c_te_f,o_te_f);

NumTrees = 100;
B = TreeBagger(NumTrees,X_train,Y_train(:,1),'OOBPrediction','on');
% ooberr = 1 - oobError(B,'Mode','ensemble');
Y_e = str2double(predict(B,X_test));
ACC = 1 - confusion(Y_test(:,1)',Y_e');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [train_psd,test_psd]= getPSD(data,train_data_len,sample_len)
fs = 1000;
nFB = 1;
% data = data(1:8,1:10*fs);
data_filted = bp40_FB1(double(data)',fs,nFB);

window = hanning(fs);
noverlap = 0.75*fs;
nfft = fs;
range = 'onesided';
data_len = size(data,2);
if 2*train_data_len > data_len
    error('训练数据过长')
end
w_l = sample_len;
n = 0;
for i = 1:w_l*0.25:train_data_len-w_l
    n = n + 1;
    p = pwelch(data_filted(i:i+w_l,:),window,noverlap,nfft,fs,range);
    p = p';
    train_psd(:,:,n) = p(:,1:41);
    p = pwelch(data_filted(i+train_data_len:i+w_l+train_data_len,:),...
        window,noverlap,nfft,fs,range);
    p = p';
    test_psd(:,:,n) = p(:,1:41);
end
train_psd = log(train_psd);
test_psd = log(test_psd);




function feature = getFeature(psd)

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
% size(feature)
feature = permute(feature,[2,1,3]);
% shape = size(feature)
feature = reshape(feature,size(feature,1),[]);


function [feature,label]=getLabel(c_feature,o_feature)
label_len = min(size(c_feature,1), size(o_feature,1));
feature = [c_feature(1:label_len,:); o_feature(1:label_len,:)];
label = [zeros(label_len,1); ones(label_len,1)];
label(label(:,1) == 0,2) = 1;












