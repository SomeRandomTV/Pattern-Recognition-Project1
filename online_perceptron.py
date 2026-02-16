import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Online perceptron, used for one_versus_rest

# seed i think
rng = np.random.default_rng()


class OnlinePerceptron:
    
    def __init__(self, data: pd.DataFrame, pos_class: str, learning_rate: float = 0.1, features: list = None):
        
        self.data = data
        self.learning_rate = learning_rate
        
        if features is None:
            features = ["meas_1", "meas_2", "meas_3", "meas_4"]
        self.inputs = np.array(data[features])
        self.labels = np.array(np.where(data["species"] == pos_class, 1, 0))
        
        self.num_samples, self.num_features = self.inputs.shape
        self.weights: np.ndarray = np.random.randn(self.num_features)   # create random weights
        self.bias = rng.random() # create random bias
        
        self.epochs: int = 100
        self.iterations = self.num_samples
        
    @staticmethod    
    def _activation_function(weighted_sum: float):
        
        if weighted_sum < 0:
            return 0
        return 1
    
    @staticmethod
    def _calculate_weighted_sum(weights: np.ndarray, inputs: np.ndarray, bias: float):
        
        return np.dot(weights, inputs) + bias
    @staticmethod
    def _error(predicted_label, truth_label):
        
        return truth_label - predicted_label
    
    def fit(self):
        
        rho = self.learning_rate
        features = (self.inputs)
        truth_labels = self.labels
        bias = self.bias
        weights = self.weights
        
        for epoch in range(self.epochs):
            
            print(f"=============== Epoch: {epoch} ===============")
            misclassifications = 0
            
            for i in range(self.iterations):
                
                x_i = features[i]
                t_i = truth_labels[i]
                
                weigthed_sum = self._calculate_weighted_sum(weights=weights, inputs=x_i, bias=bias)
                
                activation = self._activation_function(weighted_sum=weigthed_sum)
                
                err = self._error(predicted_label=activation, truth_label=t_i)
                if err != 0:
                    misclassifications += 1
                    
                weights = weights + rho * err * x_i
                bias = bias + rho * err
                
            self.weights = weights
            self.bias = bias
            
            print(f"======= Weights + Bias after Epoch {epoch} ===========")
            print(f" -- Weights: {self.weights}")
            print(f" -- Bias: {self.bias}")
            print(f" -- Misclassifications: {misclassifications}")
            
            self.plot_decision_boundary(title=f"Epoch {epoch}")
            
            if misclassifications == 0:
                print(f"Convergence achieved at epoch {epoch}")
                print(f"Misclassifications: {misclassifications}")
                break
    
    def plot_decision_boundary(self, xlabel: str = "Feature 1", ylabel: str = "Feature 2",
                               title: str = "Online Perceptron Decision Boundary"):
        
        # Always use first two columns of the input array
        feature_x = 0
        feature_y = 1
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        pos_mask = self.labels == 1
        neg_mask = self.labels == 0
        
        ax.scatter(self.inputs[pos_mask, feature_x], self.inputs[pos_mask, feature_y],
                   c="blue", marker="o", label="Positive Class", edgecolors="k", alpha=0.7)
        ax.scatter(self.inputs[neg_mask, feature_x], self.inputs[neg_mask, feature_y],
                   c="red", marker="x", label="Negative Class", alpha=0.7)
        
        # Decision boundary: w[0]*x + w[1]*y + bias = 0
        w_x = self.weights[feature_x]
        w_y = self.weights[feature_y]
        
        if w_y != 0:
            x_min, x_max = self.inputs[:, feature_x].min() - 0.5, self.inputs[:, feature_x].max() + 0.5
            x_vals = np.linspace(x_min, x_max, 200)
            y_vals = -(w_x * x_vals + self.bias) / w_y
            ax.plot(x_vals, y_vals, "g--", linewidth=2, label="Decision Boundary")
            
            y_min = self.inputs[:, feature_y].min() - 0.5
            y_max = self.inputs[:, feature_y].max() + 0.5
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()