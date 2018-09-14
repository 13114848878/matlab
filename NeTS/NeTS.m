classdef NeTS
    %NeTS class is a neural network that using back propagation to
    %train. The default activation function is sigmoid in hide and
    %output layer. The default cost function is quadratic function.
    %Tips: 
    %      If hide-layer number larger than 1,it may be affect 
    %      the derivative of weight and bias at front layer. I suport
    %      one hide-layer or you can use next version 'NeTS2'

    
    properties
        lr = 0.1
        batch_number = 10
        max_iter = 10000
        accept_precision = 10^(-8)
        cost_type = 'Quadratic'
        weights
        bias
        activation
        layer_number
        C
        p
        layer_depth
        
    end
    
    methods
        %initial
        function obj = NeTS(layer_number)
            obj.layer_number = layer_number;
            obj.layer_depth = length(layer_number);
            obj.weights = cell(1,obj.layer_depth-1);
            obj.bias = cell(1,obj.layer_depth-1);
            for i = 1:obj.layer_depth-1
                obj.weights{i} = randn(obj.layer_number(i),obj.layer_number(i+1));
                obj.bias{i} = randn(obj.layer_number(i+1),1);
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %train
        function obj = train(obj,X,Y)
            obj.p = randperm(size(Y,2));
            
            for i = 1:obj.max_iter
                [obj,Y_batch] = ForwardPropagation(obj,X,Y);
                obj = BackPropagation(obj,Y_batch);
                cost = CostFun(obj,Y_batch,obj.cost_type);
                obj.C(i) = cost;
                
                if cost < obj.accept_precision
                    disp('The precision is acceptable')
                    return
                end
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Cross Validation
        function ACC = CV(obj,X,Y)
            %Sample number
            ns = size(Y,2);
            %fold number
            K = 10;
            cv = cvpartition(ns,'KFold',K);
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
            
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %test
        function Y_e = test(obj,X)
            obj.activation{1} = X;
            for l = 1:obj.layer_depth-1
                z = bsxfun(@plus,obj.weights{l}'*obj.activation{l},obj.bias{l});
                obj.activation{l+1} = obj.activation_fun(z);
            end
            Y_e = obj.activation{end};
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Forward Propagation
        function [obj,Y_batch] = ForwardPropagation(obj,X,Y)
            
            [X_batch,Y_batch] = minbatch(obj,X,Y);
            obj.activation{1} = X_batch;
            for l = 1:obj.layer_depth-1
                z = bsxfun(@plus,obj.weights{l}'*obj.activation{l},obj.bias{l});
                obj.activation{l+1} = obj.activation_fun(z);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Back Propagation
        function obj = BackPropagation(obj,Y_batch)
            
            %Derivative of cost function
            %shape:(n,ns),n is neuron number n_s is sample number
            C_d = obj.activation{end} - Y_batch;
            delta = C_d.*obj.activation_fun_derivative(obj.activation{end});
            for l = 1:obj.layer_depth-1
                %that is amazing(awesome)
                d_w = obj.activation{end-l}*delta';
                d_b = delta*ones(obj.batch_number,1);
                %update error:'delta'
                delta = obj.weights{end-l+1}*delta.*obj.activation_fun_derivative(obj.activation{end-l});
                %update weight and bias
                obj.weights{end-l+1} = obj.weights{end-l+1} - (obj.lr/obj.batch_number)*d_w;
                obj.bias{end-l+1} = obj.bias{end-l+1} - (obj.lr/obj.batch_number)*d_b;
            end
            
        end%
        
        %Cost Fuction
        function C = CostFun(obj,Y_batch,type)
            
            switch type
                case 'Quadratic'
                    err = Y_batch - obj.activation{end};
                    C = 0.5*norm(err,'fro').^2;
                    
                case 'CrossEntropy'
                    %correspond last activation function is sigmoid
                    C = -( Y_batch*log(obj.activation{end})' + ...
                        (1 - Y_batch)*log(1 - obj.activation{end})' );
            end
            
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %minibatch
        function [X_batch,Y_batch] = minbatch(obj,X,Y)
            % batch_number = 10;
            s = randi( (length(obj.p)-obj.batch_number) ,1);
            e = s + obj.batch_number-1;
            X_batch = X(:,obj.p(s:e));
            Y_batch = Y(:,obj.p(s:e));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %activation function
        function a = activation_fun(~,z)
            a = logsig(z);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % derivative of activation function
        function a_d = activation_fun_derivative(~,a)
            a_d = a.*(1-a);
        end
        
        
        
        
        
        
        
        
        
        
    end
    
end


