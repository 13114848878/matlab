function fatigue_level()
time = 4;%s
fs = 1000;
dataserver = DataServer('Neuracle',9,'127.0.0.1',8712,fs,time);
dataserver.Open();
data = dataserver.GetBufferData();
dataserver.Close();
for i = 1:10
    
    data = dataserver.GetBufferData();
   
    pause(1)
    
%     subplot(2,2,[3,4])
%     plot(data(1,:)')
    %     data = get_data();
    
    feature = feature_extraction(data);
    
    
    class = model(feature);
   
    
    visualization(class,i)
    
    disp('end')
    
    %toc
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function feature = feature_extraction(data)

type = 'NN';
if strcmp(type,'reg')
    psd = data_preprocessing(data);
    
    psd_shape = size(psd);
    n_ch = psd_shape(1);
    bands = {1:5,6:8,9:13,14:31};
    n_band = length(bands);
    frequency = 0:30;
    f_mat = repmat(frequency,n_ch,1);
    
    i = 0;
    for band = 1:n_band
        i = i + 1;
        band_mean = squeeze(mean(psd(:,bands{band}),2));
        feature(:,i) = band_mean;
        p_band = psd(:,bands{band});
        f_band = f_mat(:,bands{band});
        p_band_sum = squeeze(sum(p_band,2));
        cgf_band = squeeze(sum(p_band.*f_band,2))./p_band_sum;
        a = squeeze(sum(p_band.*f_band.^2,2));
        c = squeeze(cgf_band).^2;
        b = c./p_band_sum;
        fv_band = (a - b)./p_band_sum;
        i = i + 1;
        feature(:,i) = squeeze(cgf_band);
        i = i + 1;
        feature(:,i) = squeeze(fv_band);
    end
    feature(:,i+1) = squeeze(sum(psd(:,1:41,:),2));
    k = size(feature,2);
    n_f = k;
    for i = 1:n_f-1;
        for j = i+1:n_f;
            k = k+1;
            feature(:,k) = feature(:,j)./feature(:,i);
        end
    end
    feature = reshape(feature,[],1);
    
elseif strcmp(type,'NN')
    tic
    psd = data_preprocessing(data);
    toc
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
    feature = reshape(feature,[],1);
    %feature
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function psd = data_preprocessing(data)
% %filter
fs = 1000;
% fs2 = 100;
% fp2 = 45;
% fp1 = 2;
% fs1 = 0.1;
% Wp = [fp1,fp2]/fs/2;
% Ws = [fs1,fs2]/fs/2;
% Rp = 3;
% Rs = 60;
% [n,~] = buttord(Wp,Ws,Rp,Rs);
% hd = design(fdesign.bandpass('N,F3dB1,F3dB2',n+1,fp1,fp2,fs),'butter');

window = hanning(fs);
noverlap = [];
nfft = fs;
range = 'onesided';
nFB = 1;
data_filted = bp40_FB1(data(1:8,1:end)',fs,nFB);
% data_filted = filter(hd,data(1:8,:)');
%size(data_filted)
% subplot(2,2,[3,4])
% plot(data_filted(:,1))
% xlabel('Sample Point')
% ylabel('Amplitude(μV)')

%由于滤波器问题，前3s的数据不能用
p = pwelch(data_filted,window,noverlap,nfft,fs,range);
p = p';
psd = log(p(:,1:41));
%size(psd)
% subplot(2,2,2)
% plot(psd')
% xlim([1,45])
% xlabel('Frequency(Hz)')
% ylabel('PSD')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function class = model(feature)
load('net')
[Y_e,~] = net.test(feature,[0,1]');
[max_num,class] = max(Y_e);
%class
if isnan(max_num)
    disp('input is zeros')
end


%
% n_f = length(feature);
% if feature(randi(n_f)) > feature(randi(n_f))
%     class = 1;
% else
%     class = 0;
% end

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
%title('Green mean alert, red mean fatigue')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function data = get_data()
% 
% dataserver = DataServer('Neuracle',9,'127.0.0.1',8712,1000,10);
% dataserver.Open();
% dataserver.GetBufferData;
% data = dataserver.GetBufferData;
% pause(0.1)
% plot(data(2,:))

















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [y]=band_pass(x,fs,fs2,fp2,fp1,fs1)
% %该函数采用blackman窗实现带通滤波
% %x为输入信号，fs，为采样频率
% %fs2,fp2分别为阻带上截至频率和通带上截至频率
% %fp1，fs1分别为通带下截止频率和阻带下截至频率
% %ps：输入时以上四个滤波参数按从大到小输入即可
% %20150615 by boat
% 
% 
% %求对应角频率
% ws2=fs2*2*pi/fs;
% wp2=fp2*2*pi/fs;
% wp1=fp1*2*pi/fs;
% ws1=fs1*2*pi/fs;
% 
% 
% %求滤波器的阶数
% B=min(ws2-wp2,wp1-ws1);   %求过渡带宽
% % N=ceil(12*pi/B);
% N=ceil(11*pi/B)+1;
% 
% 
% %计算滤波器系数
% wc2=(ws2+wp2)/2;
% wc1=(ws1+wp1)/2;
% wp=[wc1,wc2];
% hn=fir1(N-1,wp,'bandpass',blackman(N));
% y=filter(hn,1,x);




