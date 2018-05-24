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
