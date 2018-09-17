%使用滤波代替FFT提取特征

function train_ui2

tips = 'Please choose an option';
close_start = 0;
open_start = 0;
collection_start = 0;
% train = 0;


figure('pos',[300 300 700 300],'menu','none')
htext = uicontrol('pos',[100 100 200 100],'style','text','string',tips);
uicontrol('pos',[10  10 100 20],'string','start','callback',@collect_data);
uicontrol('pos',[120 10 100 20],'string','stop' ,'callback',@stop);
uicontrol('pos',[230 10 100 20],'string','close eyes' ,'callback',@close_eye);
uicontrol('pos',[340 10 100 20],'string','open eyes' ,'callback',@open_eye);
% uicontrol('pos',[450 10 100 20],'string','Train model' ,'callback',@train_model);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function close_eye(~,~)
        close_start = 1;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function open_eye(~,~)
        open_start = 1;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function stop(~,~)
        collection_start = 0;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function train_model
%         train = 1;
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function collect_data(~,~)
        collection_start = 1;
        set(htext,'string','Connected')
        %         pause(1)
        time = 15;%s
        fs = 1000;
        dataserver = DataServer('Neuracle',9,'127.0.0.1',8712,fs,time);
        dataserver.Open();
        %         data = dataserver.GetBufferData();
        c = 0;
        data_close = [];
        data_open = [];
        not_close_data = 1;
        not_open_data = 1;
        have_data = 0;
        
        while c < 2 && collection_start
            data = dataserver.GetBufferData();
            
            pause(0.01)
            plot(data(1,:))
            if close_start
                disp('close eyes')
                set(htext,'string','Collecting')
                %                 pause(1)
                isTriggerBox = true;
                if isTriggerBox
                    triggerBox = TriggerBox();
                end
                close_start_maker = 1;
                triggerBox.OutputEventData(uint8(close_start_maker));
                pause(10)
                close_end_marker = 2;
                triggerBox.OutputEventData(uint8(close_end_marker));
                set(htext,'string','Collection is done')
                close_start = 0;
            end
            
            if open_start
                set(htext,'string','Collecting')
                isTriggerBox = true;
                if isTriggerBox
                    triggerBox = TriggerBox();
                end
                
                open_start_maker = 3;
                triggerBox.OutputEventData(uint8(open_start_maker));
                pause(10)
                open_end_marker = 4;
                triggerBox.OutputEventData(uint8(open_end_marker));
                set(htext,'string','Collection is done')
                open_start = 0;
            end
            
            ismaker = nonzeros(data(9,:));
            
            if length(ismaker) == 2
                %get data
                data_m = data;
                if ismaker(1) == 1 && ismaker(2) == 2 && not_close_data
                    
                    s = find(data_m(9,:)==1);
                    e = find(data_m(9,:)==2);
                    data_close = data_m(1:8,s:e);
                    %                     size(data_close)
                    c = c + 1;
                    disp('收集到闭眼数据')
                    not_close_data = 0;
                end
                
                if ismaker(1) == 3 && ismaker(2) == 4 && not_open_data
                    s = find(data_m(9,:)==3);
                    e = find(data_m(9,:)==4);
                    data_open = data_m(1:8,s:e);
                    c = c + 1;
                    disp('收集到睁眼数据')
                    not_open_data = 0;
                end
                
            end%if
            
        end%while
        
        if sum(nonzeros(data_close))>10 && sum(nonzeros(data_open))>10
            set(htext,'string','All collection is done')
            have_data = 1;
        else
            set(htext,'string','Not collected data')
        end
        dataserver.Close();
        
        %traning
        if have_data
            training(data_close,data_open)
            set(htext,'string','Model was trained')
        end
        
    end%fun


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function training(data_close,data_open)
        
        close_feature = filter_data(data_close);
        open_feature = filter_data(data_open);
        
        [feature,label] = create_label(close_feature,open_feature);
        classifier = 'MRF';
        switch classifier
            case 'RF'
                %the eyes open label is 1
                label = label(:,1);
                NumTrees = 100;
                B = TreeBagger(NumTrees,feature,label,'OOBPrediction','on');
                save B B
            case 'MRF'
                %the eyes open label is 1
                label = label(:,1);
                NumTrees = 100;
                B1 = TreeBagger(NumTrees,feature,label,'OOBPrediction','on');
                B2 = TreeBagger(NumTrees,feature,label,'OOBPrediction','on');
                B3 = TreeBagger(NumTrees,feature,label,'OOBPrediction','on');
                save B1 B1
                save B2 B2
                save B3 B3
        end
        
        
    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function feature = filter_data(data)
        
        fs = 1000;
        data_len = size(data,2);
        %window length 样本长度
        w_l = 0.5*fs;
        n = 0;
        for i = 1:w_l*0.25:data_len-w_l
            n = n + 1;
            subdata = data(:,i:i+w_l);
            for nf = 1:4
                data_filted(:,:,nf) = bp40_FB1(double(subdata)',fs,nf);
            end
            
            %是否取对数？
            band_power = squeeze(mean(data_filted.^2));%8,4
            relative_power = bsxfun(@rdivide,band_power,sum(band_power,2));
            feature(n,:) = reshape(relative_power,1,[]);
            
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [feature,label] = create_label(close_feature,open_feature)
        label_len = min(size(close_feature,1), size(open_feature,1));
        feature = [close_feature(1:label_len,:); open_feature(1:label_len,:)];
        %eyes open label is 1
        label = [zeros(label_len,1); ones(label_len,1)];
        label(label(:,1) == 0,2) = 1;
    end

end




