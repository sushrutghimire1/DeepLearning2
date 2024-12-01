# Loss/Loss.py
import numpy as np

import numpy as np
class CrossEntropyLoss:
        def __init__(self):
            self.prediction_tensor = None

        #Forward propagation
        def forward(self, prediction_tensor, label_tensor):
            epsilon = np.finfo(float).eps
            prediction_tensor = np.clip(prediction_tensor, epsilon, 1. - epsilon)
            self.prediction_tensor = prediction_tensor
            loss = -np.sum(label_tensor * np.log(prediction_tensor))
            return loss

        #Backward propagation
        def backward(self, label_tensor):
            epsilon = np.finfo(float).eps
            predictions = self.prediction_tensor
            error_tensor = -label_tensor / (predictions + epsilon)
            return error_tensor
