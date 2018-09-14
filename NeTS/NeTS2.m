classdef NeTS2
    %NeTS class is a neural network that using back propagation to
    %train. The default activation function is sigmoid in hide and
    %output layer. The default cost function is cross entropy.
    %Tips:
    %      If hide-layer number larger than 1,it may be affect
    %      the derivative of weight and bias at front layer. I suport
    %      one hide-layer or you can use next version 'NeTS2'
    %properties
    %'lr'            learning rate default: 0.1
    %'cost_type'     control the type of cost function
    %                'CrossEntropy'|Crossentropy as cost function (defalt)
    %                'Quadratic'   |Quadratic function as cost function
    %'initial_type'  'Smaller' |weigths~N(0,1/sqrt(n)),bias~N(0,1):
    %                          |n is the nueron number of front layer
    %                'Larger'  |weigths~N(0,1),bias~N(0,1)
    %'Lmabda'        The regularization coefficient
    %                regular_Cost = Cost + Lambda*0.5*||w||^2
    %b_v,w_v         the velocity of bias and weight respectively
    %alpha           the momentum coefficient
    %max_epoch
    
    properties
        lr = 1
        batch_number = 10
        accept_precision = 0
        cost_type = 'CrossEntropy'
        initial_type = 'Glorot'
        Lambda = 0
        alpha = 0
        max_epoch = 1500
        early_stopping = 0
        
        b_v
        w_v
        activation_type
        weights
        bias
        activation
        layer_number
        C
        p
        layer_depth
        ns
        
    end
    
    methods
        %initial
        function obj = NeTS2(layer_number)
            obj.layer_number = layer_number;
            obj.layer_depth = length(layer_number);
            obj = initial(obj);
            for i = 1:obj.layer_depth-2
                obj.activation_type{i} = 'tanh';
            end
            obj.activation_type{i+1} = 'softmax';
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = initial(obj)
            obj.weights = cell(1,obj.layer_depth-1);
            obj.bias = cell(1,obj.layer_depth-1);
            switch obj.initial_type
                case 'Larger'
                    for i = 1:obj.layer_depth-1
                        obj.weights{i} = randn(obj.layer_number(i), obj.layer_number(i+1));
                        obj.bias{i} = randn(obj.layer_number(i+1),1);
                        obj.w_v{i} = zeros(obj.layer_number(i), obj.layer_number(i+1));
                        obj.b_v{i} = zeros(obj.layer_number(i+1),1);
                    end
                case 'Smaller'
                    for i = 1:obj.layer_depth-1
                        obj.weights{i} = randn(obj.layer_number(i),obj.layer_number(i+1))/sqrt(obj.layer_number(i));
                        obj.bias{i} = randn(obj.layer_number(i+1),1);
                        obj.w_v{i} = zeros(obj.layer_number(i), obj.layer_number(i+1));
                        obj.b_v{i} = zeros(obj.layer_number(i+1),1);
                    end
                case 'Glorot'
                    for i = 1:obj.layer_depth-1
                        obj.weights{i} = rand(obj.layer_number(i),obj.layer_number(i+1))...
                            *2*sqrt( 6/(obj.layer_number(i) + obj.layer_number(i+1)) ) - ...
                            sqrt( 6/(obj.layer_number(i) + obj.layer_number(i+1)) );
                        obj.bias{i} = randn(obj.layer_number(i+1),1);
                        obj.w_v{i} = zeros(obj.layer_number(i), obj.layer_number(i+1));
                        obj.b_v{i} = zeros(obj.layer_number(i+1),1);
                    end
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %train
        function obj = train(obj,X,Y)
            %sample number
            obj.ns = size(Y,2);
            obj.p = randperm(obj.ns);
            max_iter = floor(obj.ns/obj.batch_number);
            for i = 1:max_iter
                [X_batch,Y_batch] = minbatch(obj,X,Y,i);
                obj = ForwardPropagation(obj,X_batch);
                obj = BackPropagation(obj,Y_batch);
                cost = CostFun(obj,Y_batch);
                obj.C(i) = cost;
                
                if cost < obj.accept_precision
                    disp('The precision is acceptable')
                    return
                end
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function best_obj = train2(obj,X,Y)
            
            criterion = 'Loss'
            
            n = size(Y,2);%sample number
            cv = cvpartition(n,'HoldOut',0.15);
            Y_train = Y(:,cv.training);
            Y_val = Y(:,cv.test);
            X_train = X(:,cv.training);
            X_val = X(:,cv.test);
            best_ACC = 0;
            best_loss = inf;
            max_fail_epoch = 100;
            C_train_all = zeros(1,obj.max_epoch);
            C_val_all = zeros(1,obj.max_epoch);
            ACC_train_all = zeros(1,obj.max_epoch);
            ACC_val_all = zeros(1,obj.max_epoch);
            
            lr_max = 0.5;
            lr_min = 0.01;
            lr_range = lr_max - lr_min;
            for i = 1:obj.max_epoch
                %linear attenuation
                learn_rate = lr_max - i*lr_range/obj.max_epoch;
                obj.lr = learn_rate;
                
                obj = train(obj,X_train,Y_train);
                [Y_train_e,C_train] = test(obj,X_train,Y_train);
                [Y_val_e,C_val] = test(obj,X_val,Y_val);
                ACC_val = 1 - confusion(Y_val,Y_val_e);
                ACC_train = 1 - confusion(Y_train,Y_train_e);
                
                if strcmp(criterion,'ACC')
                    if obj.early_stopping && (best_ACC < ACC_val)
                        best_index = i;
                        best_ACC = ACC_val;
                        best_obj = obj;
                    end
                elseif strcmp(criterion,'Loss')
                    if obj.early_stopping && (best_loss > C_val)
                        best_index = i;
                        best_loss = C_val;
                        best_ACC = ACC_val;
                        best_obj = obj;
                    end
                end
                
                if ( i - best_index ) > max_fail_epoch
                    return
                end
                
                show = 1;
                if show
                    C_train_all(i) = C_train;
                    C_val_all(i) = C_val;
                    ACC_train_all(i) = ACC_train;
                    ACC_val_all(i) = ACC_val;
                    epoch_show(obj,C_train_all,C_val_all,ACC_train_all,ACC_val_all,i,best_ACC,best_index)
                end
                
                
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Cross Validation
        function ACC = CV(obj,X,Y)
            
            %fold number
            K = 10;
            cv = cvpartition(obj.ns,'KFold',K);
            ACC = zeros(1,K);
            for i = 1:K
                X_train = X(:,cv.training(i));
                X_test = X(:,cv.test(i));
                Y_train = Y(:,cv.training(i));
                Y_test = Y(:,cv.test(i));
                obj = train(obj,X_train,Y_train);
                Y_e = test(obj,X_test);
                ACC(i) = 1 -confusion(Y_test,Y_e);
            end
            ACC = mean(ACC);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %test
        function [Y_e,C] = test(obj,X,Y)
            obj.activation{1} = X;
            for l = 1:obj.layer_depth-1
                z = bsxfun(@plus,obj.weights{l}'*obj.activation{l},obj.bias{l});
                obj.activation{l+1} = obj.activation_fun(z,obj.activation_type{l});
            end
            Y_e = obj.activation{end};
            C = CostFun(obj,Y);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Forward Propagation
        function obj = ForwardPropagation(obj,X_batch)
            obj.activation{1} = X_batch;
            for l = 1:obj.layer_depth-1
                z = bsxfun(@plus,obj.weights{l}'*obj.activation{l},obj.bias{l});
                obj.activation{l+1} = obj.activation_fun(z,obj.activation_type{l});
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Back Propagation
        function obj = BackPropagation(obj,Y_batch)
            
            %Derivative of cost function
            %shape:(n,ns),n is neuron number n_s is sample number
            delta = delta0(obj,Y_batch);
            
            for l = 1:obj.layer_depth-1
                %that is amazing(awesome)
                d_w = obj.activation{end-l}*delta';
                d_b = delta*ones(obj.batch_number,1);
                %update error:'delta'
                if l < obj.layer_depth-1
                    delta = obj.weights{end-l+1}*delta.*obj.activation_fun_derivative(obj.activation{end-l},obj.activation_type{end-l});
                end
                %update weight and bias.
                %Lambda is regularization coefficient
                w_g = - (obj.lr/obj.batch_number)*d_w;%gradient of weight
                b_g = - (obj.lr/obj.batch_number)*d_b;%gradient of bias
                obj.weights{end-l+1} = (1-obj.Lambda*obj.lr/obj.ns)*obj.weights{end-l+1}...
                    + w_g + (obj.alpha*obj.w_v{end-l+1});%momentum
                obj.bias{end-l+1} = obj.bias{end-l+1} + b_g + (obj.alpha*obj.b_v{end-l+1});
                %update momentum
                obj.w_v{end-l+1} = w_g;
                obj.b_v{end-l+1} = b_g;
            end
            
        end%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Cost Fuction
        function C = CostFun(obj,Y_batch)
            
            switch obj.cost_type
                case 'Quadratic'
                    n = size(Y_batch,2);
                    C = 0.5*norm(Y_batch - obj.activation{end}, 'fro').^2/n;
                    
                case 'CrossEntropy'
                    %check the activation function of output layer
                    switch obj.activation_type{end}
                        case 'sigmoid'
                            CE = -( Y_batch.*log(obj.activation{end}) + ...
                                (1 - Y_batch).*log(1 - obj.activation{end}));
                            CE(isnan(CE)) = 0;
                            C = mean(CE(:));
                        case 'softmax'
                            CE = -Y_batch.*log(obj.activation{end});
                            CE(isnan(CE)) = 0;
                            C = mean(sum(CE));
                    end
                    %                     CE(isnan(CE)) = 0;
                    %                     C = mean(CE(:));
            end
            
            penalty = 0;
            if obj.Lambda > 0
                for i = 1:obj.layer_depth-1
                    penalty = penalty + norm(obj.weights{i}(:))^2;
                end
            end
            
            C = C + (0.5*obj.Lambda/obj.ns)*penalty;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function delta = delta0(obj,Y_batch)
            
            switch obj.cost_type
                case 'CrossEntropy'
                    delta = obj.activation{end} - Y_batch;
                case 'Quadratic'
                    C_d = obj.activation{end} - Y_batch;
                    delta = C_d.*obj.activation_fun_derivative(obj.activation{end},obj.activation_type{end});
            end
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %minibatch
        function [X_batch,Y_batch] = minbatch(obj,X,Y,i)
            % batch_number = 10;
            e = i*obj.batch_number;
            s = e - obj.batch_number + 1;
            X_batch = X(:,obj.p(s:e));
            Y_batch = Y(:,obj.p(s:e));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %activation function
        function a = activation_fun(~,z,type)
            switch type
                case 'sigmoid'
                    a = logsig(z);
                case 'tanh'
                    a = tanh(z);
                case 'softmax'
                    a = softmax(z);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % derivative of activation function
        function a_d = activation_fun_derivative(~,a,type)
            switch type
                case 'sigmoid'
                    a_d = a.*(1-a);
                case 'tanh'
                    a_d = 1-a.^2;
            end
            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function epoch_show(~,C_train_all,C_val_all,ACC_train_all,ACC_val_all,i,best_ACC,best_index)
            if i == 1
                figure;
            end
            C_train_all = C_train_all(1:i);
            C_val_all = C_val_all(1:i);
            ACC_train_all = ACC_train_all(1:i);
            ACC_val_all = ACC_val_all(1:i);
            
            pause(0.01)
            subplot(212)
            plot(ACC_train_all,'r')
            hold on
            plot(ACC_val_all,'b')
            hold off
            legend({'Train','Test'},'Location','best')
            xlabel('Training Epoch')
            ylabel('Accuracy')
            if i>10
                title({['Accuracy mean(last 10):',num2str(mean(ACC_val_all(end-10:end)),'%.3f'),'(test) ',...
                    num2str(mean(ACC_train_all(end-10:end)),'%.3f'),'(train)'];...
                    ['Best Val ACC:',num2str(best_ACC,'%.3f'),' at epoch:',num2str(best_index)]})
            end
            grid on
            %     title('max')
            subplot(211)
            plot(C_train_all,'r')
            hold on
            plot(C_val_all,'b')
            hold off
            %     xlabel('Training Epoch')
            ylabel('Loss')
            legend({'Train','Test'},'Location','best')
            if i > 10
                title(['Loss mean(last 10):',num2str(mean(C_val_all(end-10:end)),'%.3f'),'(test) ',...
                    num2str(mean(C_train_all(end-10:end)),'%.3f'),'(train)'])
            end
            grid on
            
            
        end
        
        
        
        
    end
    
end


