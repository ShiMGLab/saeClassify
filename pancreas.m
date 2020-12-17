clc; 
clear all;
close all;
load('/Users/apple/Desktop/single.cell/data/GSE84133_RAW/pancreas.train.cbind.mat')
load('/Users/apple/Desktop/single.cell/data/GSE84133_RAW/pancreas.test.cbind.mat')

% training samples labels
num = size(train_label,1);
label = [];
ls= train_label(:,2);
for j = 1:num
    if ls{j,:}== 'acinar'
        label(j) = 1;
    elseif ls{j,:}== 'activated_stellate'
        label(j) = 2;
    elseif ls{j,:}== 'alpha'
        label(j) = 3;
    elseif ls{j,:}== 'beta'
        label(j) = 4;
    elseif ls{j,:}== 'delta'
        label(j) = 5;
    elseif ls{j,:}== 'ductal'
        label(j) = 6;
    elseif ls{j,:}== 'endothelial'
        label(j) = 7;
    elseif ls{j,:}== 'epsilon'
        label(j) = 8;
    elseif ls{j,:}== 'gamma'
        label(j) = 9;
    elseif ls{j,:}== 'macrophage'
        label(j) = 10;
    elseif ls{j,:}== 'mast'
        label(j) = 11;
    elseif ls{j,:}== 'quiescent_stellate'
        label(j) = 12;
    elseif ls{j,:}== 't_cell'
        label(j) = 13;
    end
end
train_y = label;


% test samples labels
num = size(test_label,1);
label = [];
ls= test_label(:,2);
for j = 1:num
    if ls{j,:}== 'acinar'
        label(j) = 1;
    elseif ls{j,:}== 'activated_stellate'
        label(j) = 2;
    elseif ls{j,:}== 'alpha'
        label(j) = 3;
    elseif ls{j,:}== 'beta'
        label(j) = 4;
    elseif ls{j,:}== 'delta'
        label(j) = 5;
    elseif ls{j,:}== 'ductal'
        label(j) = 6;
    elseif ls{j,:}== 'endothelial'
        label(j) = 7;
    elseif ls{j,:}== 'epsilon'
        label(j) = 8;
    elseif ls{j,:}== 'gamma'
        label(j) = 9;
    elseif ls{j,:}== 'macrophage'
        label(j) = 10;
    elseif ls{j,:}== 'mast'
        label(j) = 11;
    elseif ls{j,:}== 'quiescent_stellate'
        label(j) = 12;
    elseif ls{j,:}== 't_cell'
        label(j) = 13;
    end
end
test_y = label;

%% STEP 1: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; 

hiddenSize = 300;     % number of hidden units, this parameters could be tuned.
sparsityParam = 0.01;% desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter    
beta = 3;            % weight of sparsity penalty term       

%%======================================================================
%% STEP 2: data pancreas input
train_data_L = train_data;
test_data_L = test_data;

%normalization of training samples  
train_data=(train_data_L-repmat(mean(train_data_L,2),1,size(train_data_L,2)))./repmat(std(train_data_L,0,2),1,size(train_data_L,2));

%normalization of test samples  
test_data=(test_data_L-repmat(mean(test_data_L,2),1,size(test_data_L,2)))./repmat(std(test_data_L,0,2),1,size(test_data_L,2));

%%======================================================================
%% STEP 3: Implement sparseAutoencoderCost
%
%  We suggest implementing the sparseAutoencoderCost function using the following steps:
%
%  (a) Implement forward propagation in your neural network, and implement the 
%      squared error term of the cost function.  Implement backpropagation to 
%      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
%      to verify that the calculations corresponding to the squared error cost 
%      term are correct.
%
%  (b) train_z2grad and test_z2grad are dimensional matrix for analysis.

train_data_c = [train_data,test_data];
train_data_c= train_data_c(all(~isnan(train_data_c),2),:);

% number of input units of training samples
train_visibleSize = size(train_data_c,1);  

%  Obtain random parameters theta of training samples
train_theta = initializeParameters(hiddenSize, train_visibleSize);

[train_cost, train_grad, train_z2grad, train_a2grad] = sparseAutoencoderCost(train_theta, train_visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, train_data_c);

                                 
 %% STEP 4: multi-class SVM for cell type identification
%training dataset
train_x = train_z2grad(:,1:1802);
%test dataset
test_x = train_z2grad(:,1803:3604);

%%  5-fold cv
datas_normal = train_x';
labels = train_y';

k =5;
sum_accuracy_svm = 0;
sum_AUC_svm = 0;
[m,n] = size(datas_normal);
indices = crossvalind('Kfold',m,k);
    
for i = 1:k
    test_indic = (indices == i);
    train_indic = ~test_indic;
    train_datas = datas_normal(train_indic,:);
    train_labels = labels(train_indic,:);
    test_datas = datas_normal(test_indic,:);
    test_labels = labels(test_indic,:);
    
    % fitcsvm two types£¬fitcecoc multiple types 
    t = templateSVM('Standardize',true,'KernelFunction','linear');
    classifer = fitcecoc(train_datas,train_labels,'Learners',t);
    
    predict_label  = predict(classifer, test_datas);
    accuracy_svm = length(find(predict_label == test_labels))/length(test_labels);%×¼È·ÂÊ
    sum_accuracy_svm = sum_accuracy_svm + accuracy_svm;
      
end
 
%mean accuracy
mean_accuracy_svm = sum_accuracy_svm / k;

disp('mean accuracy=');   
disp( mean_accuracy_svm)


%% independetn test

t = templateSVM('Standardize',true,'KernelFunction','linear');
classifer = fitcecoc(train_x',train_y','Learners',t);

predict_test_label  = predict(classifer, test_x');

accuracy_test_svm = length(find(predict_test_label == test_y'))/length(test_y');
disp('test accuracy=');   
disp( accuracy_test_svm)

%% figure

figure;
subplot(121);
group_test = test_y;
color = [0 0 0;1 0 0;0 1 0;0 0 1;1 1 0;1 0 1;0 1 1;0.667 0.667 1;1 0.5 0;0.5 0 0;0.5 0.5 0.5;1 0.667 0.667;0.5 0 0.5];
gscatter(test_x(1,:),test_x(2,:),group_test,color,'.',12);
%gscatter(test_x(1,:),test_x(2,:),group_test,'rkgbymcw','.',12);

title('real distribution');
xlabel('SAE-1');
ylabel('SAE-2');
legend('Location','Northwest','FontSize',12);
grid on;

subplot(122);
group_test = predict_test_label;
gscatter(test_x(1,:),test_x(2,:),group_test,color,'.',12);
%gscatter(test_x(1,:),test_x(2,:),group_test,'rkgb','.',12);

title('predicted distribution');
xlabel('SAE-1');
ylabel('SAE-2');
legend('Location','Northwest','FontSize',12);
grid on;




