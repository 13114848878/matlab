function test_1
    
    flag = true;
    k    = 0;
    
    figure('pos',[500 500 500 500],'menu','none')
    htext = uicontrol('pos',[100 100 100 100],'style','text','string',k);
    uicontrol('pos',[10  10 100 20],'string','start','callback',@call1);
    uicontrol('pos',[120 10 100 20],'string','stop' ,'callback',@call2);
    
    function call1(~, ~)
        flag = true;
        while flag
            k = k + 1;
            set(htext,'string',k)
            drawnow
        end
    end
    
    function call2(~, ~)
        flag = false;
    end
    
end