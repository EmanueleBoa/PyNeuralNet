# PyNeuralNet

PyNeuralNet is a python library to implement Artificial Neural Networks for classification and regression with a few lines of code and it is thought to be as easy to use as possible.

It consists of two modules:
* `layer.py` where the different types of layer are defined.
* `neuralnet.py` where the object `NeuralNet` (feed-forward neural network) and all the main methods are defined. 

At the moment, only feed-forward neural networks with full connectivity between layers can be implemented.

## 1. Building your neural network
This chapter will guide you to build your own neural network with the `NeuralNet` class.

### The `build_network` method
The easiest way of building a structured neural network is by using the method `build_network`.

Here is an example:

`from neuralnet import NeuralNet`

`net = NeuralNet()`

`net.build_network(2,4,4,1, hidden_type="Tanh", out_type="Linear", scope="Regression", verbose=True)`

In such a way you create a network for regression called `net` and composed of an input layer of dimension 2, two hidden layers of dimension 4 and hyperbolic tangent activation function, and an output layer of dimension 1 and linear activation function.

As shown in the example, the first argument taken by the method `build_network` is a list of integers which define the dimensions of the layers. The second argument is a list of optional keyworded variables, described in the following:
* `hidden_type`: string that specifies the type of hidden layers (see section Types of layer below). If not specified is set to a default value.
* `output_type`: string that specifies the type of output layers (see section Types of layer below). If not specified is set to a default value.
* `scope`: can be `'Regression'` or `'Classification'`. If not specified is set to `'Classification'` by default. 
* `verbose`: boolean. If `True` the structure of the network is printed on screen. If not specified, it is set to `True` by default. 

The only limitation of the method `build_network` is that hidden layers will all be of the same type. If you want to have hidden layers of different types you can use the the alternative method described below.

### Adding layers to a network
Another, more general, way of building the same neural network of the previous example is the following:

`from neuralnet import NeuralNet`

`net = NeuralNet()`

`net.add_layer("Input",2)`

`net.add_layer("Tanh",4)`

`net.add_layer("Tanh",4)`

`net.add_layer("Linear",1)`

`net.set_scope("Regression")`

The function `add_layer(type, dim)` adds a layer to the network. The type is specified by the string `type`, and the dimension by an integer `dim`. When using this method for building the network, it is also necessary to specify the scope of the network with the method `set_scope`. The structure of the network can be printed on screen at the moment of execution by adding the line 

`net.print_network_structure()`

### Types of layer
When building a network you can choose different types of layers. The difference between the types is essentially in the activation function which is used.

The possible types of layer are:
* `'Input'`: input layer. Specified only when using the method `add_layer`.
* `'Linear`: linear layer with identity activation function.
* `'Sigmoid'`: layer with Sigmoid activation function.
* `'Tanh'`: layer with hyperbolic tangent activation function. 
* `'SoftSign'`: layer with SoftSign activation function.
* `'ReLU'`: layer with ReLU activation function.
* `'Softmax'`: layer with Softmax activation function. Used for classification problems as output layer.

When created, weights and biases of the layers are initialized with the normalized Xavier initialization. 

## 2. Training and validation of the network

This chapter will guide you to train and validate your neural network.

### Data set and target output
For training and validating the network you need to build a data set consisting of examples of input vectors with known output. Specifically, you will need a list (or a numpy array) containing the examples, and a list (or numpy array) with the corresponding target outputs. Suppose we call the first list `dataset` an the second one `target`. They must be built such that `target[i]` is the target output of the example `dataset[i]`.

For a classification problem, the target output is simply a label indicating the class to which the input vector belongs. It must be an integer between `0` and `c-1`, where `c` is the total number of classes you wish to distinguish.

For a regression problem, the target output is simply the value of the function you wish to approximate corresponding to the given example.  

### Training the network
The training consists in optimizing the parameters of the networks based on the information contained in a training set of data. This is done by iteratively minimizing an error function with a minimization algorithm. 

The class `NeuralNet` uses either the _sum of squares_ error function or the _cross-entropy_ error function, depending on the type of the output layer and the scope of the network. Two stochastic minimization algorithms are implemented: Stochastic Gradient Descent (SGD) and Adam [Kingma, Diederik, and Jimmy Ba. “Adam: A method for stochastic optimization.” arXiv preprint arXiv:1412.6980 (2014)]. 

#### Setting the parameters for the training
You can set the parameters for the training with the method `set_training_param`. The argument taken by this method is a list of optional keyworded variables:
* `solver`: is the algorithm for parameters optimization. You can choose between `"sgd"` (Stochastic Gradient Descent) or `"adam"`. If not specified, it is set to `"sgd"` by default. 
* `learning_rate`: the learning rate used in the optimization process. If not specified, it is set to `0.01` by default. 
* `momentum`: a positive number between 0 and 1. If specified, a momentum term is used. Only used when solver is `"sgd"`.
* `weight_decay`: a positive number. If specified, a weight decay (L2 penalty regularization term) is used. If not specified, it is set to `0` by default.
* `beta1`: exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). If not specified, it is set to `0.9` by default. Only used when solver is `"adam"`.
* `beta2`: exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). If not specified, it is set to `0.99` by default. Only used when solver is `"adam"`.
* `epsilon`: value for numerical stability in adam. If not specified, it is set to `1e-8` by default. Only used when solver is `"adam"`.
* `batchsize`: an integer indicating the size of the batches when using a batched version of sgd or adam.
If not specified, it is set to `1` by default.
* `training_rounds`: an integer indicating the number of times the data set is used for the training. If not specified, it is set to `1` by default.
* `return_error`: boolean. If `True` a list with the error during the training is returned by the method that performs the training. If not specified, it is set to `True` by default.

Here is an example of how to use the method:

`net.set_training_param(solver="sgd", learning_rate=0.1, momentum=0.9, return_error=True, batchsize=10, training_rounds=10)`

#### Parameters choice
The optimal choice of the parameters depends on the specific problem. You should be particularly careful when choosing the learning rate. If this parameter is too low, then the learning is going to be very slow, while if it is too large, divergent oscillations (and overflows) may result. 

#### Training
The training is performed using the method `trainOnDataset(dataset, target)`, which takes as inputs a list of examples, `dataset`, and a list of target outputs, `target`. 

Here is an example of how to use the method:

`net.set_training_param(learning_rate=0.5, momentum=0.9, return_error=True, batchsize=10, training_rounds=10)`

`train_error = net.trainOnDataset(training_set, training_target)`

In this example `return_error=True` and, hence, the method `trainOnDataset` returns a list with the error.

### Validation of the network (for classification only)
Validating the network consists in evaluating its performance on a new set of examples, usually called _validation set_. In a classification problem, the performance, or accuracy, is measured as the percentage of correctly classified examples. 

The methods presented in the following are meant to be used only for classification problems.

#### Validation
The validation of the network can be performed with the method `validate`, which takes as inputs a list of examples and a list of target outputs, and returns the score, i.e. the percentage of correctly classified examples.

Here is an example of how to use the method:

`validation_score = net.validate(validation_set, validation_target)`

#### Training and validation with only one date set
If you have only one data set, you can split it into two separate data sets, one for the training and one for the validation of the network.

This can be done with the method `trainAndValidate(dataset, target, fraction)` which takes as inputs a data set and its corresponding target outputs, randomly splits it into a training set and a validation set, and performs the training and the validation of the network. The method returns a list with the error during the training (if `return_error` is set to `True`), the score on the training set, and the score of the validation set.

Here is an example of how to use the method:

`net.set_training_param(learning_rate=0.5, return_error=True)`

`train_error, training_score, validation_score = net.trainAndValidate(dataset, targets, 0.9)`

In this example the given data set is randomly split so that 90% of its data is used for the training, and the remaining 10% for the validation. 

#### Cross-validation
The best way to asses the accuracy of the network when having only one data set available is to use the cross-validation procedure. This procedure consists in first randomly dividing the data set into N distinct segments. Then, the network is trained using data from N-1 of the segments and its performance is tested using the remaining segment. This procedure is repeated for each of the N possible choices for the segment which is omitted from the training process, and the accuracy averaged over all N results.

Cross-validation can be performed with the method `cross_validation`.

Here is an example of how to use the method:

`average_score = net.cross_validation(dataset, target, N)`

## 3. Make predictions with the network

This chapter will guide you to use your trained network in order to make predictions on new data.

### Predict the output of new data
The output of a new input vector can be produced with the method `activate`. This method takes as input a vector and returns the output of the network. 

Here is an example of how to use the method:

`output = net.activate(input_vector)`

This method is thought mainly for regression problems. Nonetheless, in the case of a classification problem, the method will return a list with the probabilities that the input vector belongs to any of the c classes.

To produce the output of many vectors contained in a data set you can use the method `predict_dataset`, which takes as input a data set and returns a list with the corresponding outputs.

Here is an example of how to use the method:

`output_list = net.predict_dataset(dataset)`


### Classify new data
To classify a new input vector you can use the method `classify`, which takes as input a vector and returns a label indicating to most likely class to which it belongs.

Here is an example of how to use the method:

`output_label = net.classify(input_vector)`

To classify many vectors contained in a data set you can use the method `classify_dataset`, which takes as input a data set and returns a list of labels.

Here is an example of how to use the method:

`output_labels = net.classify_dataset(dataset)`

### Save and load a trained network
You can save your trained network on a file with the method `save`.

Here is an example of how to use the method:

`net.save('net.txt')`

In the example, the network is saved on a file named `net.txt`.

You can load a saved network with the method `load`.

Here is an example of how to use the method:

`net = NeuralNet()`

`net.load('net.txt')`

In the example, the network saved on the file named `net.txt` is loaded.

## 4. Examples

Look at the examples to better understand how to use the module `neuralnet.py`.

In the folder _examples_ you can find two examples of neural networks used for regression and two for classification.

###  Regression
* `ex_gaussian1d.py` is the implementation of a neural network that learns to approximate a gaussian in one dimension.

* `ex_gaussian2d.py` is the implementation of a neural network that learns to approximate a gaussian in two dimensions.

###  Classification
* `ex_XOR.py` is the implementation of a neural network that learns to solve the XOR problem.

* `mnist_recognition.py` is the implementation of a neural network that learns to classify handwritten digits (numbers between 0 and 9) from the MNIST Data set (available at http://yann.lecun.com/exdb/mnist/). To run this program it is important that the data set and the module `readDataset.py` are in the same folder as the program.

All the examples make use of the modules `neuralnet.py` and `layer.py`, which must be in the same folder as the examples in order to be imported (or you need to import them with the right path).
