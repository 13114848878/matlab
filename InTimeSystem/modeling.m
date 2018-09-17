clear;clc
X_name = 'feature_10s';
Y_name = 'label_10s';

load (['C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system\',X_name])
load (['C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system\',Y_name])

eval(['X = double(',X_name,');']);
eval(['Y = ',Y_name,';']);
X = X';
Y = Y';
ns = size(Y,2);
cv = cvpartition(ns,'HoldOut',0.15);
Y_train = Y(:,cv.training);
Y_test = Y(:,cv.test);
X_train = X(:,cv.training);
X_test = X(:,cv.test);

hide_layer_num = [3,3];
layer_number = [size(X,1),hide_layer_num,size(Y,1)];
% net1 = myNN(X,Y,layer_number);

net = NeTS2(layer_number);
% net.max_iter = 10000;
% net.lr = 0.05;
% net.alpha = 0.1;%momentum coefficient
net.batch_number = 10;
% net.cost_type = 'Quadratic';
% initial_type = 'Larger';
% net.Lambda = 0.1;%regularization coefficient
% net.accept_precision = 0;
% figure
net.early_stopping = 1;
net = net.train2(X,Y);
net_1s = net;
save net_1s net_1s



