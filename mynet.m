classdef mynet
    %MYNET 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        weights
        bias
        activation
        layer_number
        C
        lr = 0.00001
        batch_number = 10        
        max_iter = 10000
        p
        layer_depth
        accept_precision = 10^(-3)
    end
    
    methods
        %initial
        function obj = mynet(layer_number)
            obj.layer_number = layer_number;
            obj.layer_depth = length(layer_number);                                   
            obj.weights = cell(1,obj.layer_depth-1);
            obj.bias = cell(1,obj.layer_depth-1);
            for i = 1:obj.layer_depth-1
                obj.weights{i} = randn(obj.layer_number(i),obj.layer_number(i+1));
                obj.bias{i} = randn(obj.layer_number(i+1),1);
            end
        end
        
        function initial(obj)
                       
            obj.weights = cell(1,obj.layer_depth-1);
            obj.bias = cell(1,obj.layer_depth-1);
            for i = 1:obj.layer_depth-1
                obj.weights{i} = randn(obj.layer_number(i),obj.layer_number(i+1));
                obj.bias{i} = randn(obj.layer_number(i+1),1);
            end
        end
        %train
        function obj = train(obj,X,Y)
            obj.p = randperm(size(Y,2));
            for i = 1:obj.max_iter
                [obj,Y_batch] = ForwardPropagation(obj,X,Y);
                obj = BackPropagation(obj,Y_batch);
                cost = CostFun(obj,Y_batch,'L2');
                obj.C(i) = cost;
                if cost < obj.accept_precision
                    disp('The precision is acceptable')
                    return
                end
            end
        end
        
        %test
        function Y = test(obj,X)
            obj.activation{1} = X;
            for l = 1:obj.layer_depth-1
                z = bsxfun(@plus,obj.weights{l}'*obj.activation{l},obj.bias{l});
                obj.activation{l+1} = mysigmoid(z);
            end
            Y = obj.activation{end};
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Forward Propagation
        function [obj,Y_batch] = ForwardPropagation(obj,X,Y)
            
            [X_batch,Y_batch] = minbatch(obj,X,Y);
            obj.activation{1} = X_batch;
            for l = 1:obj.layer_depth-1
                z = bsxfun(@plus,obj.weights{l}'*obj.activation{l},obj.bias{l});
                obj.activation{l+1} = mysigmoid(z);
            end
        end
        
        function obj = BackPropagation(obj,Y_batch)
            %Back Propagation
            %Derivative of cost function
            %shape:(n,ns),n is neuron number n_s is sample number
            C_d = obj.activation{end} - Y_batch;
            delta{1} = C_d.*mysigmiod_derivative(obj.activation{end});
            for l = 1:obj.layer_depth-1
                %that is amazing(awesome)
                d_w = obj.activation{end-l}*delta{l}';
                d_b = delta{l}*ones(obj.batch_number,1);
                delta{l+1} = obj.weights{end-l+1}*delta{l}.*mysigmiod_derivative(obj.activation{end-l});
                %update weight
                obj.weights{end-l+1} = obj.weights{end-l+1} + obj.lr*d_w;
                obj.bias{end-l+1} = obj.bias{end-l+1} + obj.lr*d_b;
            end
            
        end%
        
        %Cost Fuction
        function C = CostFun(obj,Y_batch,type)
            switch type
                case 'L2'
                    C = 0.5*norm(Y_batch - obj.activation{end}).^2;
                case 'CrossEntropy'
                    C = -Y_batch*log(obj.activation{end})';
                               
            end
            
            
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %minibatch
        function [X_batch,Y_batch] = minbatch(obj,X,Y)
            % batch_number = 10;
            s = randi((length(obj.p)-obj.batch_number),1);
            e = s + obj.batch_number-1;
            X_batch = X(:,obj.p(s:e));
            Y_batch = Y(:,obj.p(s:e));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %sigmoid function
        function a = mysigmoid(z)
            a = sigmf(z,[1,0]);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % derivative of sigmoid function
        function a_d = mysigmiod_derivative(a)
            a_d = a.*(1-a);
        end
        
        
        
        
        
        
        
        
        
        
    end
    
end


