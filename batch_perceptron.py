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
        
        return truth_label - predicted_label
    
    def fit(self, verbose: bool = False):
        
        iterations = self.iterations
        epochs = self.epochs
        rho = self.learning_rate
        inputs = (self.inputs)
        truth_labels = self.labels
        
        for epoch in range(epochs):
            
            delta_w = np.zeros(self.num_features)
            delta_b = 0
            
            for i in range(iterations):
                
                x_i = inputs[i]
                t_i = truth_labels[i]
                
                weighted_sum = self._calculated_weighted_sum(weights=self.weights, inputs=x_i, bias=self.bias)
                
                activation = self._activation_function(weighted_sum=weighted_sum)
                
                err = self._error(predicted_label=activation, truth_label=t_i)
                
                if err != 0:
                    delta_w += rho * x_i * err
                    delta_b += rho * err
                
                
            self.weights += delta_w
            self.bias += delta_b
            
            if verbose:
                print(f"Epoch {epoch}: delta_w = {delta_w}, delta_b = {delta_b}")

            if np.all(delta_w == 0) and delta_b == 0:
                print(f"Converged after {epoch + 1} epochs")
                break
        else:
            print(f"Did not converge after {epochs} epochs")
            
            
    def predict(self, new_input: np.ndarray):
        
        weighted_sm = self._calculated_weighted_sum(weights=self.weights, inputs=new_input, bias=self.bias)
        
        return self._activation_function(weighted_sum=weighted_sm)
    
    def accuracy(self):
        
        correct = 0
        for i in range(self.num_samples):
            prediction = self.predict(self.inputs[i])
            if prediction == self.labels[i]:
                correct += 1
        
        acc = correct / self.num_samples
        print(f"Accuracy: {correct}/{self.num_samples} = {acc:.4f}")
        return acc
    
    def plot_decision_boundary(self, xlabel: str = "Feature 1", ylabel: str = "Feature 2",
                               title: str = "Batch Perceptron Decision Boundary"):
        
        if self.num_features != 2:
            print(f"Warning: Cannot plot decision boundary with {self.num_features} features. Skipping plot.")
            return
        
        feature_x = 0
        feature_y = 1
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        pos_mask = self.labels == 1
        neg_mask = self.labels == 0
        
        ax.scatter(self.inputs[pos_mask, feature_x], self.inputs[pos_mask, feature_y],
                   c="blue", marker="o", label="Positive Class", edgecolors="k", alpha=0.7)
        ax.scatter(self.inputs[neg_mask, feature_x], self.inputs[neg_mask, feature_y],
                   c="red", marker="x", label="Negative Class", alpha=0.7)
        
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
