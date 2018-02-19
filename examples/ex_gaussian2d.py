"""
Example: A Neural Network to approximate
a gaussian in 2 dimensions (Regression)
"""

import numpy as np
import math as m
import random as r
import layer
import sys
from neuralnet import NeuralNet


"""
Create training set with corresponding target
"""
training_set = []
training_target = []

for i in range(10000):
    x = r.uniform(-5.0, 5.0)
    y = r.uniform(-5.0, 5.0)
    vector = [x,y]
    z = m.exp(-0.5*(x*x+y*y))/2.0*m.pi
    training_set.append(vector)
    training_target.append(z)
training_set = np.array(training_set)
training_target = np.array(training_target)


"""
Create validation set with corresponding target
"""
validation_set = []
validation_target = []

for i in range(100):
    x = -5.0+i*0.1
    for j in range(100):
        y = -5.0+j*0.1
        vector = [x,y]
        z = m.exp(-0.5*(x*x+y*y))/2.0*m.pi
        validation_set.append(vector)
        validation_target.append(z)
validation_set = np.array(validation_set)
validation_target = np.array(validation_target)

"""
Build Neural network
"""
net = NeuralNet()
net.build_network(2,8,8,1, hidden_type="Sigmoid", out_type="Linear", scope="Regression", verbose=True)

"""
Another way of building the same network
"""
#net = NeuralNet()
#net.add_layer("Input",2)
#net.add_layer("Sigmoid",8)
#net.add_layer("Sigmoid",8)
#net.add_layer("Linear",1)
#net.set_scope("Regression")
#net.print_network_structure()

"""
Training
"""
# set parameters for training
net.set_training_param(learning_rate=0.5, momentum=0.9, return_error=True, batchsize=10, training_rounds=10)
# train
train_error = net.trainOnDataset(training_set, training_target)

"""
Print error during training on file
"""
file_out = open("train_error.dat",'w')
for i in range(len(train_error)):
    file_out.write(str(i+1)+" "+str(train_error[i])+"\n")
file_out.close()

"""
Make prediction on validation set
"""
out = net.predict_dataset(validation_set)

"""
Print neural network prediction and
target output for validation set on a file
"""
file_out = open("results.dat",'w')
for i in range(len(out)):
    file_out.write(str(validation_set[i][0])+" "+str(validation_set[i][1])+" "+str(out[i])+\
    " "+str(validation_target[i])+"\n")
file_out.close()

"""
Save network on file for future use
"""
net.save("net.txt")

"""
Load saved network
"""
#net2 = NeuralNet()
#net2.load("net.txt")
