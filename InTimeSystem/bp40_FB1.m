function  y=bp40_FB1(x,fs,nFB)
fs=fs/2;

%h1
fls1(1)=[2]; 
fls2(1)=[0.2];
fhs1(1)=[4]; 
fhs2(1)=[100];

%h1
fls1(2)=[4]; 
fls2(2)=[0.2];
fhs1(2)=[8]; 
fhs2(2)=[100];

%h1
fls1(3)=[8]; 
fls2(3)=[0.2];
fhs1(3)=[12]; 
fhs2(3)=[100];

%h1
fls1(4)=[12]; 
fls2(4)=[0.2];
fhs1(4)=[30]; 
fhs2(4)=[100];

Wp=[fls1(nFB)/fs fhs1(nFB)/fs];%3 7
Ws=[fls2(nFB)/fs fhs2(nFB)/fs];%1 1


Wn=0;
%[N,Wn]=cheb1ord(Wp,Ws,5,30);
[N,Wn]=cheb1ord(Wp,Ws,3,40);
[B,A] = cheby1(N,0.5,Wn);
y = filtfilt(B,A,x);

 