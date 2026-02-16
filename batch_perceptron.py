import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng()


class BatchPerceptron:
    
    def __init__(self, data: pd.DataFrame, pos_class: str, learning_rate: float, features: list = None):
        
        self.data = data
        self.learning_rate = learning_rate
        
        if features is None:
            features = ["meas_1", "meas_2", "meas_3", "meas_4"]
            
        self.inputs = np.array(data[features])
        self.labels = np.array(np.where(data["species"] == pos_class, 1, 0))
        
        self.num_samples, self.num_features = self.inputs.shape
        self.weights = np.random.randn(self.num_features)
        self.bias = rng.random()
        
        self.epochs = 100
        self.iterations = self.num_samples
    
    @staticmethod    
    def _activation_function(weighted_sum: float):
        
        if weighted_sum < 0:
            return 0
        return 1
    
    @staticmethod
    def _calculated_weighted_sum(weights: np.ndarray, inputs: np.ndarray, bias: float):
        
        return np.dot(weights, inputs) + bias
    
    
    @staticmethod
    def _error(predicted_label: int, truth_label: int):
        
        return predicted_label - truth_label
    
    def fit(self):
        
        iterations = self.iterations
        epochs = self.epochs
        rho = self.learning_rate
        inputs = (self.inputs)
        truth_labels = self.labels
        bias = self.bias
        weights = self.weights
        
        for epoch in range(epochs):
            
            print(f"=============== Epoch: {epoch} ===============")
            
            misclassifications = []
            
            for i in range(iterations):
                
                x_i = inputs[i]
                t_i = truth_labels[i]
                
                weighted_sum = self._calculated_weighted_sum(weights=weights, inputs=x_i, bias=bias)
                
                activation = self._activation_function(weighted_sum=weighted_sum)
                
                err = self._error(predicted_label=activation, truth_label=t_i)
                
                if err != 0:
                    misclassifications.append(x_i)
                
                
        
        
    
    
    
    
        
        
        
        
        
        
        