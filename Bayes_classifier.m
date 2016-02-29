
close all
clear
clc

load 'mnist_mat.mat'

U_y = zeros(20,10);
Cov_y = zeros(20,200);
Pi_y = zeros(1,10);
f_array = zeros(10,1);
for y = 0:1:9
	n_y = sum(label_train == y);
	class = find(label_train == y);
	class_size=size(class,2);
	pi_y = 1/size(Xtrain,2)*class_size;
	Pi_y(y+1) = pi_y;
	u_sum = sum(Xtrain(:,class(1):class(1)+class_size-1),2);
	u_y = 1/n_y * u_sum;
	U_y(:,y+1)=u_y;
	cov_sum = zeros(20,20);

	for i =0:1:class_size-1
		cov_i = (Xtrain(:,class(1)+i)-u_y)*(Xtrain(:,class(1)+i)-u_y)';
		cov_sum = cov_sum + cov_i;
	end
	cov_y = 1/n_y * cov_sum;
	Cov_y(:,y*20+1:(y+1)*20)=cov_y;
end

PredictedLabel = zeros(500,1);
f = zeros(10,1);
for i=1:1:500
    for y=0:1:9
        u_y = U_y(:,y+1);
        cov_y = Cov_y(:,20*y+1:20*(y+1));
        pi_y = Pi_y(y+1);
    multi_test= -1/2*(Xtest(:,i)-u_y)'*cov_y^-1*(Xtest(:,i)-u_y);
    f_y = pi_y/sqrt(det(cov_y))*exp(multi_test);
    f(y+1) = f_y;
    end
    f_max = max(f);
    test_label = find(f==f_max)-1;
   PredictedLabel(i)=test_label;
end

disp('Confusion matrix:')
C = confusionmat(label_test',PredictedLabel)
disp('Prediction accuracy is:')
accuracy = trace(C)/500

for y = 0:1:9
    fprintf('Class %i mean\n',y)
	u_y = U_y(:,y+1)
	fprintf('Class %i covariance\n',y)
	cov_y = Cov_y(:,20*y+1:20*(y+1))
	fprintf('Class %i class prior\n',y)
	pi_y = Pi_y(:,y+1)
	u_img = Q*u_y;
	u_img = reshape(u_img,28,28);
	figure
	imagesc(u_img')
	title('Image of the mean of each Gaussian')
end


%Three misclassified examples
imgt1 = reshape(Q*Xtest(:,39),28,28);
imgp1 = reshape(Q*Xtest(:,251),28,28);
figure
suptitle('Misclassified example 1')
subplot(1,2,1)
imagesc(imgt1')
title('True label 0')
subplot(1,2,2)
imagesc(imgp1')
title('Predicted label 5')

imgt2 = reshape(Q*Xtest(:,85),28,28);
imgp2 = reshape(Q*Xtest(:,401),28,28);
figure
suptitle('Misclassified example 2')
subplot(1,2,1)
imagesc(imgt2')
title('True label 1')
subplot(1,2,2)
imagesc(imgp2')
title('Predicted label 8')

imgt3 = reshape(Q*Xtest(:,220),28,28);
imgp3 = reshape(Q*Xtest(:,451),28,28);
figure
suptitle('Misclassified example 3')
subplot(1,2,1)
imagesc(imgt3')
title('True label 4')
subplot(1,2,2)
imagesc(imgp3')
title('Predicted label 9')

%probability distribution on 10 digits
for y=0:9
figure
hist(PredictedLabel(y*50+1:(y+1)*50),50)
title(sprintf('Probability distribution on digit %i',y))
axis([0 9 0 50])
end
