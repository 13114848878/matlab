clear;clc;
X_name = 'feature_10s';
Y_name = 'label_10s';
net_name = 'net_1s';
name2 = 'B1';
name3 = 'B2';
homepath = 'C:\Users\bci523-2\Desktop\tiansen\可穿戴设备\Matlab程序\in time system\';
load ([homepath,X_name])
load ([homepath,Y_name])
load([homepath,net_name])
load([homepath,name2])
load([homepath,name3])

eval(['X = double(',X_name,');']);
eval(['Y = ',Y_name,';']);
eval(['model = ',net_name,';']);
eval(['model2 = ',name2,';']);
eval(['model3 = ',name3,';']);

classifier = 'MM';%'MM':multi-model
switch classifier
    case 'NN'
        X = X';
        Y = Y';
        [Y_e,~] = model.test(X,Y);
        ACC = 1 -confusion(Y,Y_e)
        
    case 'RF'
        y = Y(:,1);
        y_e = predict(model,X);
        
        ACC = 1 - confusion(y',str2double(y_e)')
        
    case 'MM'
        y = Y(:,1);
        [Y_e1,~] = model.test(X',Y');
        Y_e1 = Y_e1';
        y_e1(Y_e1(:,1)>0.5,1) = 1;
        y_e2 = str2double(predict(model2,X));
        y_e3 =str2double( predict(model3,X));
        
        y_e = mode([y_e1,y_e2,y_e3],2);
        ACC1 = 1 - confusion(y',y_e1')
        ACC2 = 1 - confusion(y',y_e2')
        ACC3 = 1 - confusion(y',y_e3')
        ACC = 1 - confusion(y',y_e')
end



