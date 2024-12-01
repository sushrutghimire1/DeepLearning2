# Layers/SoftMax.py
import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        max_input = np.max(input_tensor, axis=1, keepdims=True)
        exps = np.exp(input_tensor - max_input)
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        self.output = exps / sum_exps
        return self.output

    def backward(self, error_tensor):
        batch_size, num_classes = error_tensor.shape

        softmax_deriv = self.output[..., np.newaxis] * np.eye(num_classes) - \
                        self.output[:, np.newaxis, :] * self.output[:, :, np.newaxis]

        gradient = np.einsum('bij,bj->bi', softmax_deriv, error_tensor)

        return gradient
