X_name = 'feature_1s';
Y_name = 'label_1s';

load (['C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system\',X_name])
load (['C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system\',Y_name])

eval(['X = double(',X_name,');']);
eval(['Y = ',Y_name,';']);
Y = Y(:,1);

W = LDA(X,Y);

L = [ones(size(Y,1),1) X] * W';

% Calculate class probabilities
P = exp(L) ./ repmat(sum(exp(L),2),[1 2]);

sum(P(:,2) == Y)/length(Y)