import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(43)

class MultiClassPerceptron:
    
    def __init__(self, data: pd.DataFrame, learning_rate: float = 0.1, features: list = None):
        
        self.data = data
        if features is None:
            features = ["meas_3", "meas_4"]
        self.feaures = data[features]
        self.labels = data["species"]
        
        self.num_samples, self.num_features = self.feaures.shape
        
        self.setosa_w = np.random.randn(self.num_features)
        self.virgi_w = np.random.randn(self.num_features)
        self.versi_w = np.random.randn(self.num_features)
        
        self.setosa_b = rng.random()
        self.virgi_b = rng.random()
        self.versi_b = rng.random()