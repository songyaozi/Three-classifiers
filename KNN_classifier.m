
close all
clear
clc

load 'mnist_mat.mat'
class = zeros(500,5);
C_group = zeros(10,50);
Accuracy = zeros(1,5);
D= zeros(1,5000);
for k = 1:1:5
    for i=1:1:500
        for j = 1:1:5000
            dist = sqrt(sum((Xtest(:,i)-Xtrain(:,j)).^2));
            D(j) = dist;
        end
        [dist_min,DIndex] = sort(D);
        Trainingidx = DIndex(1:k);
        labelarray = zeros(1,k);
        for t=1:1:k
            labelarray(t)=label_train(Trainingidx(t));
        end
        label=mode(labelarray);
        class(i,k)=label;
        
    end
    
    C= confusionmat(label_test',class(:,k));
    C_group(:,(k-1)*10+1:k*10) = C;
    Accuracy(k) = trace(C)/500;
    fprintf('\n\n')
    fprintf('k=%i predicted accuracy is %f',k,Accuracy(k))
end

%show three misclassified images when k = 1
disp('The first one is the one in corresponding C(4,2), which indicating the true class is 3, and the predicted class is 1.')
disp('The second one is the one in corresponding C(8,2), which indicating the true class is 7, and the predicted class is 1.')
disp('The first one is the one in corresponding C(10,2), which indicating the true class is 9, and the predicted class is 1.')
img11t = Q*Xtest(:,194);
img11t = reshape(img11t,28,28);
img12t = Q*Xtest(:,369);
img12t = reshape(img12t,28,28);
img13t = Q*Xtest(:,456);
img13t = reshape(img13t,28,28);
img11p = Q*Xtest(:,51);
img11p = reshape(img11p,28,28);


figure
subplot(1,4,1)
imagesc(img11t')
title('True class is 3')
subplot(1,4,2)
imagesc(img12t')
title('True class is 7')
subplot(1,4,3)
imagesc(img13t')
title('True class is 9')
subplot(1,4,4)
imagesc(img11p')
title('Predicted class is 1')

%show three misclassified images when k = 3
disp('The first one is the one in corresponding C(3,7), which indicating the true class is 2, and the predicted class is 6.')
disp('The second one is the one in corresponding C(6,7), which indicating the true class is 5, and the predicted class is 6.')
disp('The first one is the one in corresponding C(10,7), which indicating the true class is 9, and the predicted class is 6.')
img31t = Q*Xtest(:,141);
img31t = reshape(img31t,28,28);
img32t = Q*Xtest(:,291);
img32t = reshape(img32t,28,28);
img33t = Q*Xtest(:,457);
img33t = reshape(img33t,28,28);
img3p = Q*Xtest(:,301);
img3p = reshape(img3p,28,28);
figure
subplot(1,4,1)
imagesc(img31t')
title('True class is 2')
subplot(1,4,2)
imagesc(img32t')
title('True class is 5')
subplot(1,4,3)
imagesc(img33t')
title('True class is 9')
subplot(1,4,4)
imagesc(img3p')
title('Predicted class is 6')

%show three misclassified images when k = 5
disp('The first one is the one in corresponding C(3,9), which indicating the true class is 2, and the predicted class is 8.')
disp('The second one is the one in corresponding C(3,9), which indicating the true class is 2, and the predicted class is 8.')
disp('The first one is the one in corresponding C(4,9), which indicating the true class is 3, and the predicted class is 8.')
img51t = Q*Xtest(:,104);
img51t = reshape(img51t,28,28);
img52t = Q*Xtest(:,110);
img52t = reshape(img52t,28,28);
img53t = Q*Xtest(:,166);
img53t = reshape(img53t,28,28);
img5p = Q*Xtest(:,401);
img5p = reshape(img5p,28,28);
figure
subplot(1,4,1)
imagesc(img51t')
title('True class is 2')
subplot(1,4,2)
imagesc(img52t')
title('True class is 2')
subplot(1,4,3)
imagesc(img53t')
title('True class is 3')
subplot(1,4,4)
imagesc(img5p')
title('Predicted class is 8')