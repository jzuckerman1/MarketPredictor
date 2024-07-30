#  
# This is my Neural Network created in a project also posted
# on my github page.
#
# import packages --------------------------------------------------------------
import numpy as np
import random
from itertools import permutations

# FOMC Reader ----------------------------------------------------------------
class Network(object):
    """
    Creates a Neural Network in specified layers with implemented Neural Network Methods
    """
    def __init__(self, sizes : list[int]):
        """
        Initializes the Network with random, normally distributed, biases and weights.

        Args:
            sizes (list): List of the number of neurons in each layer
        """
        self.sizes = sizes
        self.biases = [np.random.randn(layer, 1) for layer in sizes[1:]]
        self.weights = [np.random.randn(layer, prev) for prev, layer in zip(sizes[:-1], sizes[1:])]
    
    
    
    def feedForward(self, inputs : list[np.ndarray]):
        """
        Returns the output of the network given the input list

        Args:
            inputs (list): Input to the lyaer
        """
        for bias, weight in zip(self.biases, self.weights):
            inputs = np.tanh(np.dot(weight, inputs) + bias)
        return inputs
    
    
    
    def SGD(self, training_data : list[tuple[np.ndarray, np.ndarray]], epochs : int, mini_batch_size : int, learningRate : float, test_data : list[tuple[np.ndarray, np.ndarray]] =None) -> None:
        """
        Trains the network using Stoachastic Gradient Descent. 

        Args:
            training_data (list of tuples): Training Data for the network
            epochs (int): number of iterations to adjust network
            mini_batch_size (int): size of each batch to pass through
            learningRate (float): Amount to shift layers in backpropogration
            test_data (list, optional): Passed through test data. Defaults to None.
        """
        
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j + mini_batch_size] for j in range(0, len(training_data), mini_batch_size)]
            mini_batches = [batch for batch in mini_batches if len(batch) == mini_batch_size]
            
            for batch in mini_batches:
                self.update_mini_batch(batch, learningRate)
                
            if test_data:
                print(f'Epoch {i + 1}: {self.evaluate(test_data)} / {len(test_data)}') 
            else:
                print(f'Epoch {i + 1} complete!')
        
        
        
    def update_mini_batch(self, mini_batch : list[tuple[np.ndarray, np.ndarray]], learningRate : float):
        """
        Updates the weights and biases using SGD and Back Propogation

        Args:
            mini_batch (list[tuple[np.ndarray, np.ndarray]]): list containing a subset of the training data
            learningRate (float): Amount by which to shift existing layers after backpropogration
        """
        new_biases = [np.zeros(b.shape) for b in self.biases] #array of zeros w/ same dimension [( 1, 2), (3, 4)] -> [(0, 0), (0, 0)] 
        new_weights = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            change_new_biases, change_new_weights = self.backprop(x, y)
            new_biases = [cur + change for cur, change in zip(new_biases, change_new_biases)]
            new_weights = [cur + change for cur, change in zip(new_weights, change_new_weights)]
        
        self.weights = [weight - (learningRate / len(mini_batch) * new_weight) for weight, new_weight in zip(self.weights, new_weights)]
        self.biases = [bias - (learningRate / len(mini_batch) * new_biases) for bias, new_biases in zip(self.biases, new_biases)] 
        
        
    def backprop(self, x, y) -> tuple:
        """
        Return a tuple representing the gradient for the cost function. 

        Args:
            x (np.ndarray): Input array
            y (np.ndarray): Expected output array
        """
        new_biases = [np.zeros(b.shape) for b in self.biases] #array of zeros w/ same dimension [( 1, 2), (3, 4)] -> [(0, 0), (0, 0)] 
        new_weights = [np.zeros(w.shape) for w in self.weights]
        
        #feed forward
        activation = x
        activations = [x] #To store all activations in each layer
        zs = [] #To store all z vectors
        
        # Iterate across layers
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation).reshape(-1, 1) + bias # Have to reshape to correct dimensionality
            zs.append(z)
            activation = np.tanh(z)
            activations.append(activation)
        
        # backwards pass
        delta = self.cost_derivative(activations[-1].transpose()[0], y).reshape(-1, 1) * (1 / np.cosh(zs[-1])) ** 2 # Find the cost of our result (sech^2 from the derivative of tanh)
        
        new_biases[-1] = delta
        new_weights[-1] = np.dot(delta, activations[-2].transpose())
        
        for layer in range(-2, -len(self.sizes), -1): # Using negative indexing
            z = zs[layer]
            sp = np.tanh(z)
            delta = np.dot(self.weights[layer + 1].transpose(), delta) * sp
            new_biases[layer] = delta
            if len(self.sizes) == 3: #Strange bug I encountered
                new_weights[layer] = np.dot(delta, [activations[layer - 1]])
            else:
                new_weights[layer] = np.dot(delta, activations[layer - 1].T)
            break
            
        return (new_biases, new_weights)
    
    def evaluate(self, test_data : list[tuple[np.ndarray, np.ndarray]]):
        """
        Return the number of correct outputs

        Args:
            test_data (list[tuple[np.ndarray, np.ndarray]]): test data in the form of ([input], [output])
        """
        test_results = [(self.feedForward(x), y) for (x,y) in test_data]
        return sum([int(np.mean(abs(x - y))) < 0.5 for (x, y) in test_results])
        
    
    def cost_derivative(self, output_activations, y):
        """
        Return vector of partial derivatives of cost with respect to layer for the 

        Args:
            output_activations (_type_): what we got
            y (_type_): what the goal was
        """
        return output_activations - y
    

