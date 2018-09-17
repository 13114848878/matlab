function train_ui

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
        
        close_psd = getPSD(data_close);
        open_psd = getPSD(data_open);
        close_feature = feature_extraction(close_psd);
        open_feature = feature_extraction(open_psd);
        [feature,label] = create_label(close_feature,open_feature);
        classifier = 'RF';
        switch classifier
            case 'RF'
                %the eyes open label is 1
                label = label(:,1);
                NumTrees = 100;
                B = TreeBagger(NumTrees,feature,label,'OOBPrediction','on');
                save B B
            case 'NN'
                
        end
        
        
    end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function psd = getPSD(data)
        
        fs = 1000;
        window = hanning(fs);
        noverlap = [];
        nfft = fs;
        range = 'onesided';
        nFB = 1;
        data_filted = bp40_FB1(data(1:8,1:end)',fs,nFB);
        data_len = size(data_filted,1);
        w_l = 1*fs;
        %n_i = data_len-w_l
        n = 0;
        psd = [];
        for i = 1:w_l*0.25:data_len-w_l
            n = n + 1;
            p = pwelch(data_filted(i:i+w_l,:),window,noverlap,nfft,fs,range);
            p = p';
            psd(:,:,n) = p(:,1:41);
        end
        psd = log(psd);
%         size(psd)
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function feature = feature_extraction(psd)
        
        type = 'NN';
        switch type
            case 'NN'
                
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
                feature = permute(feature,[2,1,3]);%sample,channel,feature
                feature = reshape(feature,size(feature,1),[]);
                
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [feature,label] = create_label(close_feature,open_feature)
        label_len = min(size(close_feature,1), size(open_feature,1));
        feature = [close_feature(1:label_len,:); open_feature(1:label_len,:)];
        %eyes open label is 1
        label = [zeros(label_len,1); ones(label_len,1)];
        label(label(:,1) == 0,2) = 1;
    end

end




