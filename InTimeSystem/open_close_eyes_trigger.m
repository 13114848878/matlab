%’ˆ±’—€20∑÷÷”£¨√ø÷÷◊¥Ã¨1∑÷÷”«–ªª“ª¥Œ°£
clear;clc

tips = '+';
figure('pos',[300 300 400 300],'menu','none')
txt = uicontrol('pos',[125 75 100 100],'style','text','string',tips,'FontSize',20);
%title('Please look at the cross')
isTriggerBox = true;
if isTriggerBox
    triggerBox = TriggerBox();
end
%
open_marker = 1;
close_marker = 2;
for i = 1:20
    
    if rem(i,2) == 1
        triggerBox.OutputEventData(uint8(open_marker));
        tts('’ˆ—€')
        
    elseif rem(i,2) == 0
        triggerBox.OutputEventData(uint8(close_marker));
        tts('±’—€')
        
    end
    pause(60)
    
end


