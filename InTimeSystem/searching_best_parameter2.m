function ACC_all = searching_best_parameter2()
filepath = 'C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\data';
name = 'c.bdf';
EEG = pop_readbdf( [filepath,'\',name]);
close_data = EEG.data;
name = 'o.bdf';
EEG = pop_readbdf( [filepath,'\',name]);
open_data = EEG.data;
fs = 1000;
% sample_len = 1*fs;
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
[c_tr_f,c_te_f]= getfeature(close_data,train_data_len,sample_len);

[o_tr_f,o_te_f]= getfeature(open_data,train_data_len,sample_len);

[X_train,Y_train]=getLabel(c_tr_f,o_tr_f);
[X_test,Y_test]=getLabel(c_te_f,o_te_f);

NumTrees = 100;
type = 'Multi';
switch type
    case 'Single'
        B = TreeBagger(NumTrees,X_train,Y_train(:,1),'OOBPrediction','on');
        Y_e = str2double(predict(B,X_test));
    case 'Multi'
        B1 = TreeBagger(NumTrees,X_train,Y_train(:,1),'OOBPrediction','on');
        B2 = TreeBagger(NumTrees,X_train,Y_train(:,1),'OOBPrediction','on');
        B3 = TreeBagger(NumTrees,X_train,Y_train(:,1),'OOBPrediction','on');
        Y_e1 = str2double(predict(B1,X_test));
        Y_e2 = str2double(predict(B2,X_test));
        Y_e3 = str2double(predict(B3,X_test));
        Y_e = mode([Y_e1,Y_e2,Y_e3],2);
end

ACC = 1 - confusion(Y_test(:,1)',Y_e');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [tr_f,te_f]= getfeature(data,train_data_len,sample_len)
fs = 1000;
% nFB = 1;
% data = data(1:8,1:10*fs);
% data_filted = bp40_FB1(double(data)',fs,nFB);
data_len = size(data,2);
if 2*train_data_len > data_len
    error('训练数据过长')
end
w_l = sample_len;
n = 0;
for i = 1:w_l*0.25:train_data_len-w_l
    n = n + 1;
    subdata = data(:,i:i+w_l);
    for nf = 1:4
        data_filted(:,:,nf) = bp40_FB1(double(subdata)',fs,nf);        
    end
%     data_filted = reshape(data_filted,sample_len,[]);
%是否取对数？
    band_power = squeeze(mean(data_filted.^2));%8,4
    relative_power = bsxfun(@rdivide,band_power,sum(band_power,2));
    tr_f(n,:) = reshape(relative_power,1,[]);
    
    %test
    subdata = data(:,i+train_data_len:i+w_l+train_data_len);
    for nf = 1:4
        data_filted(:,:,nf) = bp40_FB1(double(subdata)',fs,nf);        
    end
%     data_filted = reshape(data_filted,sample_len,[]);
%是否取对数？
    band_power = squeeze(mean(data_filted.^2));%8,4
    relative_power = bsxfun(@rdivide,band_power,sum(band_power,2));
    te_f(n,:) = reshape(relative_power,1,[]);
end


function [feature,label]=getLabel(c_feature,o_feature)
label_len = min(size(c_feature,1), size(o_feature,1));
feature = [c_feature(1:label_len,:); o_feature(1:label_len,:)];
label = [zeros(label_len,1); ones(label_len,1)];
label(label(:,1) == 0,2) = 1;












