clear;clc

ACC_all = searching_best_parameter2();


plot(5:5:300,ACC_all)
xlabel('Data length(s)')
ylabel('Accuracy')
hold on
plot(ones(1,51)*50, 0:0.02:1, '--k')
plot(ones(1,51)*30, 0:0.02:1, '--k')
legend({'100','200','300','400','500','600','700','800','900','1000'})
hold off