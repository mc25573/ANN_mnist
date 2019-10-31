# ANN MNIST Classifier (from scratch)
MATLAB code for an artificial neural network that classifies digits from the [MNIST data set](http://yann.lecun.com/exdb/mnist/). This ANN can only have one hidden layer. This network is coded from scratch, meaning it does not use something like the MATLAB Deep Learning Toolbox. I did this because I wanted to understand all the math behind ANNs. Additionally, writing all the code from scratch enables me to make small changes that might be difficult when using a framework. There are very few commits to this project because I decided to add it to GitHub after I had finished it.
## Running the Code
The network can be run by simply pressing the run button (f5). I recommend using ctrl+enter to run it section by section because you might not want it to, say, make plots everytime. Another reason is there is pooling.m which replaces the DATA INITIALIZATION section of the main code. If you want to max pool the input data, run pooling.m, then run the main code (ANN_mnist.m) section by section, skipping the first section.
## Code Flow
First the code reads the data from the two csv files. One for the training data. One for the testing data. Then target matrices are made and input data is normalized and centered around zero. Next, weight matrices are created and many of the hyperparameters of the network are initialized. Finally, the network is trained using stochastic graident descent and tested. The network is considered correct if the index of the highest value of the output matches the index of the highest value of the target output. There are a few sections of code for plotting which can be interesting. The first section plots a moving average of the error while training. The next section plots the images that the network got wrong. It tells you what the image actually is and what the network guess it is with a confidence value. 
## Performance
I've been able to get as much as 98.96% images correct. Which from what I can tell, for a basic network like this, is rather good. This was with max pooling, 200 hidden neurons, and 40 epochs. However, you won't alwasy get such values with those parameters due to the inherent randomness of the model. 

Without max pooling, the network is able to get to higher accuracy with much fewer epochs, but it takes much longer to train because there are so many more parameters (784 input neurons vs. 256).

Here is the output from the network with 200 hidden neurons and 7 epochs without max pooling:

![200_7](https://github.com/mc25573/ANN_mnist/blob/master/images/200_7.JPG)

To get similar accuracy with max pooling, about 25 epochs are required, but it takes far less time:

![200_25_pooling](https://github.com/mc25573/ANN_mnist/blob/master/images/200_25_pooling.JPG)

Not only are you able to get high accuracy faster with max pooling, but it seems you are also able to get higher accuracy. I was not able to reach 98.96% accuracy without max pooling.
## Update
I added a softmax activation to the output layer in place of sigmoid. I do not believe any changes to the back prop algorithm were necessary because the derivative of the softmax is the same as the sigmoid when the indices of the inputs and outputs are equal. I am a bit confused because I'm not sure when they wouldn't be equal. 2d array? Anyway, this had a couple of effects. First, I was able to achieve a higher accuracy. Second, pooling no longer seemed to work as well. Here's the highest accuracy I was able to achieve. Notice the time spent training was about the time it took to achieve the best results with pooling:

![200_7](https://github.com/mc25573/ANN_mnist/blob/master/images/100_softmax.PNG)
