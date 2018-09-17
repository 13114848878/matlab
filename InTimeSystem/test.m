function target=test(in,out,t)
[model,k,ClassLabel]=LDATraining(in,out);
target=LDATesting(t,k,model,ClassLabel);