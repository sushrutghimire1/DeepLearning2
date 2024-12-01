# Layers/ReLU.py
import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # Gradient for ReLU is 1 for positive input and 0 for negative input
        return error_tensor * (self.input_tensor > 0)
