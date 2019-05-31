1 - Hello everyone! I have worked on a project and I want to introduce the project to you. 

2- I have implemented a neural network model to classify animal images. I selected neural networks because it is becoming more popular and I am interested in it. I used Keras which is a python framework to implement the model.
I faced some problems and dealt with them.

3- I find the dataset I use from KTH Royal University in Stockholm, Sweden. There are 19 classes in this dataset, so it is a challenging one. I get 50 percent accuracy due to some reasons, I will come to that in the next slides.

4- In the dataset there are images that are almost evenly distributed from all 19 classes. All of them are not in the same size but their sizes are close to 250 by 250 pixels.

5- These are some samples from the dataset.

6- I resized all the images to 250x250 to be able to give them to the network. I split the dataset to training and test set by using 80 by 20 ratio. I did not used a validation set because I was already decided the model that I will use, the convolutional network, and the hyperparameter number was 1, only the learning rate.

7- I used softmax activation function at the output layer because it is good for multiclass classification. Basically it produces probabilities for all the classes.

8- I used cross entropy loss function which is a good combination with softmax activation. It's formula is shown there. It's formula is minus one times true value of ith output ,that is  1 or 0, times logarithm of the ith output of the model.

9- Stochastic gradient descent, or SGD in short is the algorithm that I used to optimize the network. In this algorithm, training sample is selected randomly and samples are used one by one to train the model. The learning rate is how fast the model learns and its default value is 0.01.

10- In my first trial I used a CNN with 3 layers, and I get 40 percent accuracy. Convolution is explained in this picture. Basically, we have 2 dimensional grids, input grid, filter grid, and  output grid. The filter grid slides across the input grid and corresponding numbers are multiplied and summed. 

11- In my first trial I overfit the training set, so I considered using data augmentation because having more data is good in ML applications.

12- To reduce overfitting I used dropout layers which is a mechanism that randomly discards some units' value based on a rate parameter. 

13- I flipped the images horizontally when augmenting the data. This made the input space of the network bigger. However, I still overfit the data and the accuracy was the same so I quit using it.

14- I used learning rate decay technique to increase the test set accuracy. As the algorithm become closer to the optimal point, based on the epoch number, epoch means passing over the dataset once, I reduced the learning rate. The algorithm does not go away from the optimal point since learning rate drops. 

15- I used colab notebooks because I don't have a GPU to train the model. Neural network training takes too much time. I had to upload the dataset to colab computer using a github repository.

16- You see model loss on training and test sets on the left, and you see accuracy on the right. You can realize the effect of learning rate decay technique, the fluctuations decreases over time. 

17- After all trials, I realized that I get the best accuracy when I overfit the training data. The best accuracy I can get was 50.5 percent. In my opinion the dataset was too small and as a result the model could not perform well.

18- Thanks for listening.
