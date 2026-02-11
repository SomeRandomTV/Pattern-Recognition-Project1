import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Features:
# meas_1 => Sepal Length
# meas_2 => Sepal Width
# meas_3 => Pedal Length
# meas_4 => Pedal Width

def load_data(path_to_data):
    
    try:
    
        df = pd.read_excel(path_to_data)
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File: {path_to_data} not found") from e
    
    except Exception as e:
        raise Exception(f"Error: Uncaught exception: {e}") from e
    
    return df

class IrisClassifer:
    
    def __init__(self, data: pd.DataFrame):
        
        self.data = data
        
    def print_data(self):
        
        print("========= Basic Stuff ==========")
        print(f"DF Shape: {self.data.shape}\n")
        print(f"DF D-Types:\n {self.data.dtypes}\n")
        print(f"DF Description: \n {self.data.describe()}\n")
        
        print("\n")
        print('-'*60)
        print("\n")
        
        print("========== Fisher Iris Dataset ==========")
        print("       ---------- head ----------")
        print(self.data.head())
        
        print("       ---------- Tail ----------")
        print(self.data.tail())
        
    @staticmethod   
    def _compute_sepal_length_stats(sepal_length_col: pd.Series):
        
        s_l_min = np.min(sepal_length_col)
        s_l_max = np.max(sepal_length_col)
        s_l_mean = np.mean(sepal_length_col)
        s_l_varience = np.var(sepal_length_col)
        
        return s_l_min, s_l_max, s_l_mean, s_l_varience
    
    @staticmethod   
    def _compute_sepal_width_stats(sepal_width_col: pd.Series):
        
        s_w_min = np.min(sepal_width_col)
        s_w_max = np.max(sepal_width_col)
        s_w_mean = np.mean(sepal_width_col)
        s_w_varience = np.var(sepal_width_col)
        
        return s_w_min, s_w_max, s_w_mean, s_w_varience
    
    @staticmethod   
    def _compute_pedal_length_stats(pedal_length_col: pd.Series):
        
        p_l_min = np.min(pedal_length_col)
        p_l_max = np.max(pedal_length_col)
        p_l_mean = np.mean(pedal_length_col)
        p_l_varience = np.var(pedal_length_col)
        
        return p_l_min, p_l_max, p_l_mean, p_l_varience
    
    @staticmethod   
    def _compute_pedal_width_stats(pedal_width_col: pd.Series):
        
        p_w_min = np.min(pedal_width_col)
        p_w_max = np.max(pedal_width_col)
        p_w_mean = np.mean(pedal_width_col)
        p_w_varience = np.var(pedal_width_col)
        
        return p_w_min, p_w_max, p_w_mean, p_w_varience
        
        
    def compute_stats(self):
        
        iris_df = self.data
        
        sepal_length = iris_df["meas_1"]
        sepal_width = iris_df["meas_2"]
        pedal_length = iris_df["meas_3"]
        pedal_width = iris_df["meas_4"]
        
        s_l_min, s_l_max, s_l_mean, s_l_varience = self._compute_sepal_length_stats(sepal_length_col=sepal_length)
        s_w_min, s_w_max, s_w_mean, s_w_varience = self._compute_sepal_width_stats(sepal_width_col=sepal_width)
        p_l_min, p_l_max, p_l_mean, p_l_varience = self._compute_pedal_length_stats(pedal_length_col=pedal_length)
        p_w_min, p_w_max, p_w_mean, p_w_varience = self._compute_pedal_width_stats(pedal_width_col=pedal_width)


def main(args):
    
    path_to_xlsx=Path(args.filename)
    
    
    iris_df = load_data(path_to_data=path_to_xlsx)
    
    iris_classifier = IrisClassifer(iris_df)
    
    iris_classifier.print_data()
    iris_classifier.compute_stats()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Iris Classifier", description="Compute stats on the Iris Dataset", epilog="I think, I am tired")
    parser.add_argument(
        "--filename",
        "-f",
        required=True,
        help="Path to the Iris dataset (must be .xlsx)"
    )
    
    args = parser.parse_args()
    main(args)
    
        
    
    