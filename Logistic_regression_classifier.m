%% Logistic regression
% Use softmax function
close all
clear
clc

load 'mnist_mat.mat'

XtrainNew = zeros(21,5000);
XtestNew = zeros(21,500);
XtrainNew(21,:) = 1;
XtestNew(21,:) = 1;
XtrainNew(1:20,1:5000) = Xtrain;
XtestNew(1:20,1:500) = Xtest;

%Make all w vector to zero vector
W = zeros(21,10);
Nebla_L = zeros(21,10);
eta = 0.1/5000;
%Use gradient descent to get optimal w
for t = 1:1:1000
    Xw_sum = 0;
    for i=0:1:9
        exp1 = zeros(5000,1);
        w_i = W(:,i+1);
        for j =0:1:9
			w_j = W(:,j+1);
			power1=XtrainNew'*w_j;
			exp1 = exp1 + exp(power1);
        end
        power2=XtrainNew'*w_i;
        exp2 = exp(power2);
        soft = exp2.*exp1.^-1;
        soft_sum = XtrainNew*soft;
        x_sum = sum(XtrainNew(:,500*i+1:500*(i+1)),2);
        nebla_L = x_sum - soft_sum;
        Nebla_L(:,i+1) = nebla_L;
        Xw_sum = Xw_sum+x_sum'*w_i;
    end
    logden_sum = sum(log(exp1));
    %update W matrix for one time
    L = Xw_sum-logden_sum;
    W = W+eta*Nebla_L;
    hold on
    figure(1)
    title('Log likelihood v.s. iteration times')
    plot(t,L,'b*','markersize',3)
    xlim([0,1000]) 
    hold off
end
%start to predict
disp('Since log likelihood curve is monotonically increasing, so we choose w in the 1000-th iteration as the optimal w. Then predict Xtest.')

Soft_t=zeros(500,10);
for i=0:1:9
    exp1t = zeros(500,1);
    w_i = W(:,i+1);
    for j =0:1:9
        w_j = W(:,j+1);
        power1t=XtestNew'*w_j;
        exp1t = exp1t + exp(power1t);
    end
        
    power2t=XtestNew'*w_i;
    exp2t = exp(power2t);
    soft_t = exp2t.*exp1t.^-1;
    Soft_t(:,i+1)=soft_t;
end
PredictedLabel=zeros(500,1);
for m=1:1:500
    row_max=max(Soft_t(m,:));
    col_idx = find(Soft_t(m,:)==row_max);
    label = col_idx-1;
    PredictedLabel(m,1)=label;
end
disp('Confusion matrix:')
C=confusionmat(label_test',PredictedLabel)
disp('Prediction accuracy is:')
accuracy = trace(C)/500

%Three misclassified examples
imgt1 = reshape(Q*Xtest(:,1),28,28);
imgp1 = reshape(Q*Xtest(:,301),28,28);
figure
%suptitle('Misclassified example 1')
subplot(1,2,1)
imagesc(imgt1')
title('True label 0')
subplot(1,2,2)
imagesc(imgp1')
title('Predicted label 6')

imgt2 = reshape(Q*Xtest(:,59),28,28);
imgp2 = reshape(Q*Xtest(:,401),28,28);
figure
%suptitle('Misclassified example 2')
subplot(1,2,1)
imagesc(imgt2')
title('True label 1')
subplot(1,2,2)
imagesc(imgp2')
title('Predicted label 8')

imgt3 = reshape(Q*Xtest(:,184),28,28);
imgp3 = reshape(Q*Xtest(:,101),28,28);
figure
%suptitle('Misclassified example 3')
subplot(1,2,1)
imagesc(imgt3')
title('True label 3')
subplot(1,2,2)
imagesc(imgp3')
title('Predicted label 2')

%probability distribution on 10 digits
for y=0:9
	figure
	hist(PredictedLabel(y*50+1:(y+1)*50),50)
	title(sprintf('Probability distribution on digit %i',y))
	axis([0 9 0 50])
end