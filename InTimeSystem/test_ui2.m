function test_ui2

istest = 1;
tips = 'Please choose an option';
figure('pos',[300 300 400 300],'menu','none')
htext = uicontrol('pos',[10 50 200 100],'style','text','string',tips);
uicontrol('pos',[10  10 100 20],'string','start','callback',@test);
uicontrol('pos',[120 10 100 20],'string','stop' ,'callback',@stop_test);
uicontrol('pos',[230 10 100 20],'string','restart' ,'callback',@restart);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function stop_test(~,~)
        istest = 0;
    end
    function restart(~,~)
        istest = 1;
        test
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function test(~,~)
        
        sample_len = 0.5;%s
        fs = 1000;
        dataserver = DataServer('Neuracle',9,'127.0.0.1',8712,fs,sample_len);
        dataserver.Open();
        %data = dataserver.GetBufferData();
        
        while istest
            
            data = dataserver.GetBufferData();
            
            pause(0.5)
            plot(data(1,:))
            feature = filter_data(data(1:8,:));
            
            class = model(feature);
            
            switch class
                case 2
                    set(htext,'string','eyes closed')
                    beep
                case 1
                    set(htext,'string','eyes open')
            end
          
        end
        
        dataserver.Close();
        set(htext,'string','system stopped')
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function feature = filter_data(data)
        
        fs = 1000;
               
        for nf = 1:4
            data_filted(:,:,nf) = bp40_FB1(double(data)',fs,nf);
        end
        
        %是否取对数？
        band_power = squeeze(mean(data_filted.^2));%8,4
        relative_power = bsxfun(@rdivide,band_power,sum(band_power,2));
        feature = reshape(relative_power,1,[]);
%         size(feature)
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function class = model(feature)
        
        
        
        classifier = 'MM';%'MM':multi-model
        switch classifier
            
            case 'NN'
                
                [Y_e,~] = model.test(feature,[0,1]');
                [max_num,class] = max(Y_e);
                if isnan(max_num)
                    disp('input is zeros')
                end
                
            case 'RF'
                name = 'B';
                models = load(name);
                eval(['models = ',name,';'])
                y_e = predict(model,feature');
                if y_e == 1
                    class = 1;%eyes open
                elseif y_e == 0
                    class = 2;
                end
                
            case 'MM'
                %model number
                %feature = feature';
                mn = 3;
%                 y_e = rand(1,mn);
                for i = 1:mn
                    i_str = num2str(i);
                    name = ['B',num2str(i)];
                    mymodel = load(name);
                    eval(['mymodel = mymodel.',name,';'])
%                     eval('str2double(predict(mymodel,feature))')
                    eval(['y_e(',i_str,')=str2double(predict(mymodel,feature));'])
                    %eval(['y_e(:,',i_str,') = y_e',i_str,';'])
                end
                y_e              
                y_e = squeeze(mode(y_e,2));
                
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