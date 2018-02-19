import numpy as np
import math as m
import random

class Layer (object):
    """
    A class to represent a general layer
    """
    def __init__(self, n):
        self.n = n
        self.units = np.zeros(n)
        self.type = "Layer"

    def activate(self, inputs):
        """
        Activate layer units
        """
        self.units = inputs


class Linear (Layer):
    """
    A class to represent a linear layer
    """
    def __init__(self, n, nprevious):
        Layer.__init__(self, n)
        self.nprevious = nprevious
        self.weights = np.zeros((n, nprevious))
        self.weights_old = np.zeros((n, nprevious))
        self.biases = np.zeros(n)
        self.biases_old = np.zeros(n)
        self.delta_weights = np.zeros((n, nprevious))
        self.delta_weights_old = np.zeros((n, nprevious))
        self.delta_biases = np.zeros(n)
        self.delta_biases_old = np.zeros(n)
        self.deltas = np.zeros(n)
        self.type = "Linear"

    def xavier_init_weights(self):
        """
        Initialize weights with Xavier initialization
        """
        mean = 0.0
        sigma = m.sqrt(2.0/(self.n+self.nprevious))
        for i in range(self.n):
            for j in range(self.nprevious):
                self.weights[i,j] = random.gauss(mean, sigma)
        self.weights_old = self.weights

    def set_biases(self, biases):
        """
        Set biases of linear layer equal to array biases
        """
        self.biases = biases
        self.biases_old = biases

    def set_weights(self, weights):
        """
        Set weights of linear layer equal to array weights
        """
        self.weights = weights
        self.weights_old = weights


    def activate(self, previous_layer):
        """
        Activate layer units
        """
        self.units = np.dot(self.weights, previous_layer.units)+self.biases


    def delta_out(self, target):
        """
        Compute deltas when the layer is the output layer
        and the sum of square error function is used
        """
        self.deltas = self.units - target

    def delta(self, next_layer):
        """
        Compute deltas when the layer is an hidden layer
        """
        self.deltas = np.dot(next_layer.weights.transpose(),next_layer.deltas)

    def update_delta_weights(self, previous_layer):
        """
        Update delta wheights and biases
        """
        # delta biases
        self.delta_biases += self.deltas
        # delta weights
        self.delta_weights += np.outer(self.deltas,previous_layer.units)



    def gradient_descent(self, learning_rate, momentum, batchsize, weight_decay):
        """
        Update wheights and biases with batch gradient descent (with momentum term)
        """
        # save old weights and biases
        self.weights_old = self.weights
        self.biases_old = self.biases
        # update weights and biases
        self.delta_weights = -learning_rate*(self.delta_weights/batchsize+\
        weight_decay*self.weights)+momentum*self.delta_weights_old
        self.weights = self.weights + self.delta_weights
        self.delta_biases = -learning_rate*self.delta_biases/batchsize+momentum*self.delta_biases_old
        self.biases = self.biases + self.delta_biases
        # setting delta weights and biases to zero
        self.delta_weights_old = self.weights-self.weights_old
        self.delta_biases_old = self.biases-self.biases_old
        self.delta_weights.fill(0.0)
        self.delta_biases.fill(0.0)

    def pick_class(self):
        """
        Return class with highest probability
        """
        return np.argmax(self.units)

    def build_target(self, label):
        """
        Return array with targets for classification
        """
        target = np.zeros(self.n)
        target[label] = 1.0
        return target

    def compute_error(self, target):
        """
        Return error (sum-of-squares)
        """
        error = 0.0
        if(self.n>1):
            for i in range(self.n):
                error += (target[i]-self.units[i])*(target[i]-self.units[i])
            error *= 0.5
        else:
            error += 0.5*(target-self.units[0])*(target-self.units[0])
        return error


class Softmax (Layer):
    """
    A class to represent an element-wise layer
    with Softmax activation function
    """
    def __init__(self, n, nprevious):
        if n>1:
            Layer.__init__(self, n)
            self.lin = Linear(n, nprevious)
            self.type = "Softmax"
        else:
            print "ERROR: Softmax layer cannot have dimension 1!"
            exit(1)

    def xavier_init_weights(self):
        """
        Initialize weights with Xavier initialization
        """
        self.lin.xavier_init_weights()

    def set_biases(self, biases):
        """
        Set biases of linear layer equal to array biases
        """
        self.lin.set_biases(biases)

    def set_weights(self, weights):
        """
        Set weights of linear layer equal to array weights
        """
        self.lin.set_weights(weights)

    def activate(self, previous_layer):
        """
        Activate layer units
        """
        # activate linear layer
        self.lin.activate(previous_layer)
        # activate tanh layer
        self.units = np.exp(self.lin.units)
        norm = np.sum(self.units)
        self.units = self.units/norm

    def delta_out(self, target):
        """
        Compute deltas when the layer is the output layer
        and the cross-entropy error function is used
        """
        self.lin.deltas = self.units - target

    def update_delta_weights(self, previous_layer):
        """
        Update delta wheights and biases
        """
        self.lin.update_delta_weights(previous_layer)

    def gradient_descent(self, learning_rate, momentum, batchsize, weight_decay):
        """
        Update wheights and biases with batch gradient descent (with momentum term)
        """
        self.lin.gradient_descent(learning_rate, momentum, batchsize, weight_decay)

    def pick_class(self):
        """
        Return class with highest probability
        """
        return np.argmax(self.units)

    def build_target(self, label):
        """
        Return array with targets for classification
        """
        target = np.zeros(self.n)
        target[label] = 1.0
        return target

    def compute_error(self, target):
        """
        Return error (cross-entropy)
        """
        error = 0.0
        for i in range(self.n):
            error += float(-target[i]*np.log(self.units[i]))
        return error


class Tanh (Layer):
    """
    A class to represent an element-wise layer
    with hyperbolic tangent activation function
    """
    def __init__(self, n, nprevious):
        Layer.__init__(self, n)
        self.lin = Linear(n, nprevious)
        self.type = "Tanh"

    def xavier_init_weights(self):
        """
        Initialize weights with Xavier initialization
        """
        self.lin.xavier_init_weights()

    def set_biases(self, biases):
        """
        Set biases of linear layer equal to array biases
        """
        self.lin.set_biases(biases)

    def set_weights(self, weights):
        """
        Set weights of linear layer equal to array weights
        """
        self.lin.set_weights(weights)

    def activate(self, previous_layer):
        """
        Activate layer units
        """
        # activate linear layer
        self.lin.activate(previous_layer)
        # activate tanh layer
        self.units = np.tanh(self.lin.units)

    def delta(self, next_layer):
        """
        Compute deltas when the layer is an hidden layer
        """
        tanhprime = 1.0/(np.power(np.cosh(self.lin.units),2))
        self.lin.deltas = tanhprime*np.dot(next_layer.weights.transpose(),next_layer.deltas)

    def delta_out(self, target):
        """
        Compute deltas when the layer is the output layer
        and the sum-of-square error function is used
        """
        tanhprime = 1.0/(np.power(np.cosh(self.lin.units),2))
        self.lin.deltas = tanhprime*(self.units - target)


    def update_delta_weights(self, previous_layer):
        """
        Update delta wheights and biases
        """
        self.lin.update_delta_weights(previous_layer)

    def gradient_descent(self, learning_rate, momentum, batchsize, weight_decay):
        """
        Update wheights and biases with batch gradient descent (with momentum term)
        """
        self.lin.gradient_descent(learning_rate, momentum, batchsize, weight_decay)

    def pick_class(self):
        """
        Return class with highest probability
        """
        if self.n>1:
            return np.argmax(self.units)
        else:
            if self.units[0]>0.0:
                return 1
            else:
                return 0

    def build_target(self, label):
        """
        Return array with targets for classification
        """
        if self.n>1:
            target = np.zeros(self.n)
            target[label] = 1.0
        else:
            target = label
        return target

    def compute_error(self, target):
        """
        Return error (sum-of-squares)
        """
        error = 0.0
        if(self.n>1):
            for i in range(self.n):
                error += (target[i]-self.units[i])*(target[i]-self.units[i])
            error *= 0.5
        else:
            error += 0.5*(target-self.units[0])*(target-self.units[0])
        return error


class Sigmoid (Layer):
    """
    A class to represent an element-wise layer
    with sigmoid activation function
    """
    def __init__(self, n, nprevious):
        Layer.__init__(self, n)
        self.lin = Linear(n, nprevious)
        self.type = "Sigmoid"

    def xavier_init_weights(self):
        """
        Initialize weights with Xavier initialization
        """
        self.lin.xavier_init_weights()

    def set_biases(self, biases):
        """
        Set biases of linear layer equal to array biases
        """
        self.lin.set_biases(biases)

    def set_weights(self, weights):
        """
        Set weights of linear layer equal to array weights
        """
        self.lin.set_weights(weights)

    def activate(self, previous_layer):
        """
        Activate layer units
        """
        # activate linear layer
        self.lin.activate(previous_layer)
        # activate sigmoid layer
        self.units = 1.0/(1.0+np.exp(-self.lin.units))

    def delta(self, next_layer):
        """
        Compute deltas when the layer is an hidden layer
        """
        sigmoidprime = self.units*(1.0-self.units)
        self.lin.deltas = sigmoidprime*np.dot(next_layer.weights.transpose(),next_layer.deltas)

    def delta_out(self, target):
        """
        Compute deltas when the layer is the output layer
        and the sum-of-squares error function is used
        """
        self.lin.deltas = self.units*(1.0-self.units)*(self.units-target)

    def update_delta_weights(self, previous_layer):
        """
        Update delta wheights and biases
        """
        self.lin.update_delta_weights(previous_layer)

    def gradient_descent(self, learning_rate, momentum, batchsize, weight_decay):
        """
        Update wheights and biases with batch gradient descent (with momentum term)
        """
        self.lin.gradient_descent(learning_rate, momentum, batchsize, weight_decay)

    def pick_class(self):
        """
        Return class with highest probability
        """
        if self.n>1:
            return np.argmax(self.units)
        else:
            return np.rint(self.units[0])


    def build_target(self, label):
        """
        Return array with targets for classification
        """
        if self.n>1:
            target = np.zeros(self.n)
            target[label] = 1.0
        else:
            target = label
        return target

    def compute_error(self, target):
        """
        Return error (sum-of-squares)
        """
        error = 0.0
        if(self.n>1):
            for i in range(self.n):
                error += (target[i]-self.units[i])*(target[i]-self.units[i])
            error *= 0.5
        else:
            error += 0.5*(target-self.units[0])*(target-self.units[0])
        return error
