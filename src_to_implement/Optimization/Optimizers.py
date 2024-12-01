# Optimization/Optimizers.py
import numpy as np

#Defining SGD class
class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    # Calculate the weight update for SGD
    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor