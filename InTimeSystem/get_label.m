clear;clc

filepath = 'C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system';
c_name = 'c_feature_10s_2Hz';
o_name = 'o_feature_10s_2Hz';
load([filepath,'\',c_name,'.mat'])
load([filepath,'\',o_name,'.mat'])

eval(['c_feature = ',c_name,';'])
eval(['o_feature = ',o_name,';'])

label_len = min(size(c_feature,1), size(o_feature,1));
feature = [c_feature(1:label_len,:); o_feature(1:label_len,:)];
label = [zeros(label_len,1); ones(label_len,1)];
label(label(:,1) == 0,2) = 1;

feature_10s_2Hz = feature;
label_10s_2Hz = label;
save feature_10s_2Hz feature_10s_2Hz
save label_10s_2Hz label_10s_2Hz