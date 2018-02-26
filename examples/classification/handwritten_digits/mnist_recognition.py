"""
Example: A Neural Network to recognize handwritten digits
from the MNIST data set (available at http://yann.lecun.com/exdb/mnist/)
(Classification)

In this example a neural network with one hidden layer is used for
classifying hendwritten digits (numbers beteen 0 and 9).
With such a neural network you should get an accuracy of about 95%.
To increase the accuracy you may try to increase the dimension of
the hidden layer, adding other hidden layers, and also increassing
the number of training rounds.
"""

import numpy as np
from readDataset import read
from neuralnet import NeuralNet

"""
Create training set with corresponding target labels
"""
training_set = []
training_labels = []
dataset = list(read(dataset = "training", path = ""))
#print len(dataset)
for i in range(len(dataset)):
    label, vector = dataset[i]
    training_labels.append(label)
    training_set.append(vector.flatten()/255.0)
training_set = np.array(training_set)
training_labels = np.array(training_labels)

"""
Create validation set with corresponding target labels
"""
validation_set = []
validation_labels = []
dataset = list(read(dataset = "testing", path = ""))
#print len(dataset)
for i in range(len(dataset)):
    label, vector = dataset[i]
    validation_labels.append(label)
    validation_set.append(vector.flatten())
validation_set = np.array(validation_set)
validation_labels = np.array(validation_labels)


"""
Build Neural network
"""
net = NeuralNet()
net.build_network(784,100,10, hidden_type= "Tanh", out_type="Softmax", scope="Classification")

"""
Training
"""
# set parameters for training
net.set_training_param(learning_rate=0.01, momentum=0.9, return_error=True, batchsize=10, training_rounds=1)
# train
train_error = net.trainOnDataset(training_set, training_labels)

"""
Print error during training on file
"""
file_out = open("train_error.dat",'w')
for i in range(len(train_error)):
    file_out.write(str(i+1)+" "+str(train_error[i])+"\n")
file_out.close()

"""
Validation
"""
training_score = net.validate(training_set, training_labels)
validation_score = net.validate(validation_set, validation_labels)
print ("Training score:  "+str(training_score))
print ("Validation score:"+str(validation_score))

"""
Save network on file for future use
"""
net.save("net.txt")

"""
Load saved network
"""
#net2 = NeuralNet()
#net2.load("net.txt")
