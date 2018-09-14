function test_ui

istest = 1;
tips = 'Please choose an option';
figure('pos',[300 300 300 300],'menu','none')
htext = uicontrol('pos',[10 50 200 100],'style','text','string',tips);
uicontrol('pos',[10  10 100 20],'string','start','callback',@test);
uicontrol('pos',[120 10 100 20],'string','stop' ,'callback',@stop_test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function stop_test(~,~)
        istest = 0;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function test(~,~)
        time = 1;%s
        fs = 1000;
        dataserver = DataServer('Neuracle',9,'127.0.0.1',8712,fs,time);
        dataserver.Open();
        data = dataserver.GetBufferData();
        
        while istest
            
            data = dataserver.GetBufferData();
            
            pause(0.001)
            
            psd = data_preprocessing(data);
            
            feature = feature_extraction(psd);
            
            class = model(feature);
            
            switch class
                case 2
                    set(htext,'string','eyes closed')
                    beep
                case 1
                    set(htext,'string','eyes open')
            end
            %visualization(class,i)
            
            %             disp('end')
            
            %toc
        end
        
        dataserver.Close();
        
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                AP = squeeze(sum(psd(:,bands{1}(1):bands{end}(end)),2));
                i = 0;
                for  band = 1:n_band
                    band_psd = psd(:,bands{band});
                    band_sum = squeeze(sum(band_psd,2));
                    i = i + 1;
                    feature(:,i) = band_sum./AP;
                end
                %column vector
                feature = reshape(feature,[],1);
                %feature
                
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function psd = data_preprocessing(data)
        
        fs = 1000;
        
        
        window = hanning(fs);
        noverlap = [];
        nfft = fs;
        range = 'onesided';
        nFB = 1;
        data_filted = bp40_FB1(data(1:8,1:end)',fs,nFB);
        
        
        %由于滤波器问题，前3s的数据不能用
        p = pwelch(data_filted,window,noverlap,nfft,fs,range);
        p = p';
        psd = log(p(:,1:41));
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function class = model(feature)
        
        name = 'B';
        load(name)
        eval(['model = ',name,';'])
        
        classifier = 'RF';%'MM':multi-model
        switch classifier
            
            case 'NN'
              
              [Y_e,~] = model.test(feature,[0,1]');
              [max_num,class] = max(Y_e);
              if isnan(max_num)
                  disp('input is zeros')
              end
              
            case 'RF'

                y_e = predict(model,feature');
                if y_e == 1
                    class = 1;%eyes open
                elseif y_e == 0
                    class = 2;
                end
                
            case 'MM'

                [Y_e1,~] = model.test(X',Y');
                Y_e1 = Y_e1';
                y_e1(Y_e1(:,1)>0.5,1) = 1;
                y_e2 = str2double(predict(model2,X));
                y_e3 =str2double( predict(model3,X));
                
                y_e = mode([y_e1,y_e2,y_e3],2);
                
                if y_e == 1
                    class = 1;%eyes open
                elseif y_e == 0
                    class = 2;
                end
%                 ACC1 = 1 - confusion(y',y_e1')
%                 ACC2 = 1 - confusion(y',y_e2')
%                 ACC3 = 1 - confusion(y',y_e3')
%                 ACC = 1 - confusion(y',y_e')
        end
        
    end


    function visualization(class,i)
        %pause(0.1)
        if i == 1
            %figure
        end
        if class == 1
            %Eye open
            % subplot(2,2,1)
            % bar(1,1,'FaceColor','g','EraseMode','none')
            %title('Eye Open')
        elseif class == 2
            %Eye close
            %subplot(2,2,1)
            bar(1,1,'FaceColor','r','EraseMode','none')
            %title('Eye Closed')
            
            beep
            
        end
    end



end