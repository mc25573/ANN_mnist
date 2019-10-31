%% INITIALIZE DATA
clearvars -except tr sub
if ~exist('tr','var') %don't read file again if it already exists
    tr = csvread('train.csv', 1, 0); % read train.csv
end
if ~exist('sub','var')
    sub = csvread('mnist_test.csv'); % read test.csv
end
%1,0 in csvread is offset to not include row #1 (the labels)

%1st column is desired output for first input, 2nd col. desired for next input, etc.
targetsMNIST = diag(ones(1,10)); 
target = zeros(10,length(tr)); %initialize
targetTest = zeros(10,length(sub)); %initialize

%for making target matrix corresponding to training data
for m = 1:length(tr) 
    %so that we don't get zero index in next line. this means that 2 in tr, is a "1" image
    trIdx = tr(:,1)+1;
    %desired output; "i" and "target" need to be same size   
    target(:,m) = targetsMNIST(:,trIdx(m));  
end
%for making target matrix of test data
for m = 1:length(sub) 
    trIdx = sub(:,1)+1;
    targetTest(:,m) = targetsMNIST(:,trIdx(m));    
end

iTest = (sub(:,2:end)/255)'; %test data inputs normalized and transpose
i = (tr(:,2:end)/255)'; %training inputs 
iTest = iTest - mean(mean(iTest,2)); %center around 0
i = i - mean(mean(i,2));
%inputs: # of rows = # of input neurons; # cols = # data sets
beep
%% INITIALIZE NETWORK
clearvars -except tr sub trIdx target i iTest targetTest targetsMNIST
format compact
format long

h_num = 90;                                 %number of hidden neurons per hidden layer                          
hid_w = .25*rand(h_num,size(i,1))-1/8;       %weights, rand(2,2): [w1 w2;w3 w4;w5 w6]
out_w = .25*rand(size(target,1),h_num)-1/8;  %weights of connections from hidden to output layer
n = .4;                                      %learning rate
epoch = 35;                                   %number of times the entire test data set is trained
newHid_w = hid_w;
newOut_w = out_w;
i_size=size(i,1);
%% TRAIN
count = 1;
errorYData=zeros(1,length(i));
tic
for j=1:epoch        
    %col randomizes the order in which the inputs are fed through the ANN  
    col = randperm(length(i));   
    for k=1:length(i)
        out_w = newOut_w;        
        %forward pass
        hin = newHid_w * i(:,col(k));
        hout = sigmoid(hin);
        outin = newOut_w * hout;
        out_out = softmax(outin);
        error = .5 .* (target(:,col(k)) - out_out).^2;        
        
        %track error for plotting 
        count = count + 1;
        errorYData(count) = sum(error);
        
        % BACK PROP (Stochastic Gradient Descent)
        %output layer back pass          
        dEtot_dneto = -(target(:,col(k)) - out_out) .* sig_deriv(out_out);
        %error output neurons    
        delta = dEtot_dneto * hout'; 
        
        %new weights of output layer
        newOut_w = newOut_w - n .* delta; 

        %hidden layer back pass
        h_back = out_w' * dEtot_dneto;    
        delta_h = (h_back .* sig_deriv(hout)) * i(:,col(k))';
        
        %new weights of hidden layer
        newHid_w = newHid_w - n .* delta_h;  
    end    
end
disp('Training Time: ')
toc
beep
%% TEST
format short
correct=0;
t=0;
errorYTest=zeros(1,length(sub));
for p=1:length(sub)
    hin = newHid_w * iTest(:,p);
    hout = sigmoid(hin);
    outin = newOut_w * hout;
    out_out = softmax(outin);
    errorTest = .5 .* (targetTest(:,p) - out_out).^2;
    errorYTest(p)=sum(errorTest);
    [M,I]=max(targetTest(:,p));
    [N,U]=max(out_out);
    if(I==U)
        correct=correct+1;
    else
        t=t+1;
        wrongIdx(t)=p;
        guess(t) = U-1;
        conf(t) = round(N,2);
    end    
end
disp('Inputs: ')
disp(i_size)
disp('Hidden Neurons: ')
disp(h_num)
disp('Learning Rate: ')
disp(n)
disp('Epochs: ')
disp(epoch)
percentCorrect=correct/length(sub)*100
%% PLOT ERROR AND WEIGHTS
figure
plot(1:count,movmean(errorYData,1000))
grid on
xlabel('Iteration');
ylabel('Moving Average of Error');
hist_hid=newHid_w(:); 
hist_out=newOut_w(:);
figure
hist(hist_hid,101)
title('Input to Hidden Layer Weight Distribution')
xlabel('Weight Value')
ylabel('Frequency')
figure
hist(hist_out,101)
title('Hidden to Output Layer Weight Distribution')
xlabel('Weight Value')
ylabel('Frequency')
%% PLOT INCORRECT IMAGES
figure                                         
colormap(gray)                                  % set to grayscale
for c = 1:25                                    % preview first 25 samples
    subplot(5,5,c)                              % plot them in 5 x 5 grid
    digit = reshape(sub(wrongIdx(c), 2:end), [28,28])'; % reshape to 28 x 28 image
    imagesc(digit)                              % show the image
    pbaspect([1 1 1])                           % square aspect ratio
    title(['Actual:' num2str(sub(wrongIdx(c), 1)) ' ' 'Guess:' num2str(guess(c)) ' ' 'Prob:' num2str(conf(c))]) % show the labels
end
%% PLOT TRAINING IMAGES
figure
colormap(gray) 
for c = 1:25                                    
    subplot(5,5,c) 
    imagesc(reshape(i(:,c), [28,28])')
    pbaspect([1 1 1])
    title(['Actual: ' num2str(tr(c, 1))])
end