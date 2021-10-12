"""
Example: A Neural Network to solve
the XOR problem (Classification)
"""

import numpy as np
from neuralnet import NeuralNet

"""
Create validation and training sets with corresponding target label
"""
validation_set = [[0,0],[0,1],[1,0],[1,1]]
validation_label = [0,1,1,0]

training_set = []
training_label = []

for i in range(1000):
    training_set = training_set + validation_set
    training_label = training_label + validation_label

validation_set = np.array(validation_set)
validation_label = np.array(validation_label)
training_set = np.array(training_set)
training_label = np.array(training_label)

"""
Build Neural network
"""
net = NeuralNet()
net.build_network(2,4,1, hidden_type="tanh", out_type="sigmoid", scope="classification", verbose=True)

"""
Another way of building the same network
"""
#net = NeuralNet()
#net.add_layer("input",2)
#net.add_layer("tanh",4)
#net.add_layer("sigmoid",1)
#net.set_scope("classification")
#net.print_network_structure()

"""
Training
"""
# set parameters for training
net.set_training_param(learning_rate=0.5, momentum=0.9, return_error=True, batchsize=4)
#train
train_error = net.trainOnDataset(training_set, training_label)

"""
Print error during training on file
"""
file_out = open("train_error.dat",'w')
for i in range(len(train_error)):
    file_out.write(str(i+1)+" "+str(train_error[i])+"\n")
file_out.close()

"""
Compute accuracy (score) on training
and validation sets
"""
training_score = net.validate(training_set, training_label)
validation_score = net.validate(validation_set, validation_label)
print ("\n**********           Results              **********")
print ("Training score:     "+str(training_score))
print ("Validation score:   "+str(validation_score))

"""
Save network on file for future use
"""
net.save("net.txt")

"""
Load saved network
"""
#net2 = NeuralNet()
#net2.load("net.txt")
