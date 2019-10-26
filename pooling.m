%% INITIALIZE DATA
clearvars -except tr sub
if ~exist('tr','var') %don't read file again if it already exists
    tr = csvread('train.csv', 1, 0); % read train.csv
end
if ~exist('sub','var')
    sub = csvread('mnist_test.csv'); % read test.csv
end
%1,0 in csvread is offset to not include row #1 (the labels)

targetsMNIST = diag(ones(1,10)); % 1st column is desired output for zero image input, 2nd col. desired for 1, etc.
target = zeros(10,length(tr)); %initialize
targetTest = zeros(10,length(sub)); %initialize

for m = 1:length(tr) %for making target matrix corresponding to training data
    trIdx = tr(:,1)+1; %so that we don't get zero index in next line. this means that 2 in tr, is a 1 image
    target(:,m) = targetsMNIST(:,trIdx(m)); %desired output; "i" and "target" need to be same size    
end
for m = 1:length(sub) %for making target matrix of test data
    trIdx = sub(:,1)+1;
    targetTest(:,m) = targetsMNIST(:,trIdx(m));    
end
for k =1:length(tr)
    digit = reshape(tr(k,2:end),[28,28])';
    digit = padarray(digit,[2 2],'both');   % add padding of 2 pixels
    digit = sepblockfun(digit,[2,2],'max'); % function by Mathworks user: Matt J   
    i(:,k) = reshape(digit, [1,256])';      % flatten image into vector
end
for k =1:length(sub)
    digit = reshape(sub(k,2:end),[28,28])';
    digit = padarray(digit,[2 2],'both');
    digit = sepblockfun(digit,[2,2],'max'); % sepblockfun does max pooling in this config    
    iTest(:,k) = reshape(digit, [1,256])';
end
i = i/255;
i = i-mean(mean(i,2)); %training inputs (normalized) and transpose

iTest = iTest/255;
iTest = iTest-mean(mean(iTest,2)); %test data inputs

%inputs: # of rows = # of input neurons; # cols = # data sets
beep
