import copy
import numpy as np
from Layers.Base import BaseLayer

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer  
        self.loss = []              # stores the loss values over iterations
        self.layers = []            # Sequential list of layers
        self.data_layer = None      # holds the data provider
        self.loss_layer = None      # Loss function layer

    def forward(self):
        """
        Performs a Forward Propagation through the network.
        """
        input_tensor, self.label_tensor = self.data_layer.next()  # Fetch the next data batch
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor) 
        return input_tensor

    def backward(self):
        """
        Perform a backward propagation through the network.
        Propagate the error tensor backward through the layers in reverse order.
        """
        error_tensor = self.loss_layer.backward(self.label_tensor)  
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)  

    def append_layer(self, layer):
        """
        Add a layer to the network.
        If the layer is trainable, assign it a deep copy of the optimizer.
        """
        if hasattr(layer, 'trainable') and layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)  
        self.layers.append(layer)  

    def train(self, iterations):
        """
        Train the network for a given number of iterations.
        Perform forward and backward passes, and update the loss values.
        """
        for _ in range(iterations):
            output = self.forward()  # Forward pass
            loss = self.loss_layer.forward(output, self.label_tensor)  # Compute loss value
            self.loss.append(loss)  
            self.backward()  # Backward propagation

    def test(self, input_tensor):
        """
        Perform a test (inference) pass through the network.
        Pass the input tensor through all the layers.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # Forward pass through each layer
        return input_tensor
