import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from online_perceptron import OnlinePerceptron
from batch_perceptron import BatchPerceptron

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
        print(f"DF D-Types:\n{self.data.dtypes}\n")

        print("\n")
        print("-" * 60)
        print("\n")

        print("========== Fisher Iris Dataset ==========")
        print("       ---------- head ----------")
        print(self.data.head())

        print("       ---------- Tail ----------")
        print(self.data.tail())

        print("========== Unique Values ==========")
        print(self.data.nunique())

    @staticmethod
    def _compute_column_stats(feature_column: pd.Series):

        _min = np.min(feature_column)
        _max = np.max(feature_column)
        _mean = np.mean(feature_column)
        _varience = np.var(feature_column)

        return _min, _max, _mean, _varience

    def _compute_with_class_var(self, feature_col: str):

        classes = self.data["species"].unique()
        total_n = len(self.data)
        s_w = 0.0

        for cls in classes:
            class_data = self.data[self.data["species"] == cls][feature_col]
            p_j = len(class_data) / total_n
            class_var = np.var(class_data)
            s_w += class_var * p_j

        return s_w

    def _compute_between_class_var(self, feature_col: str):

        classes = self.data["species"].unique()
        total_mean = np.mean(self.data[feature_col])
        total_n = len(self.data)
        s_b = 0.0

        for cls in classes:
            class_data = self.data[self.data["species"] == cls][feature_col]
            mean_j = np.mean(class_data)
            p_j = len(class_data) / total_n
            s_b += p_j * (mean_j - total_mean) ** 2

        return s_b

    def compute_stats(self):

        iris_df = self.data

        sepal_length = iris_df["meas_1"]
        sepal_width = iris_df["meas_2"]
        pedal_length = iris_df["meas_3"]
        pedal_width = iris_df["meas_4"]

        # compute individual stats
        s_l_min, s_l_max, s_l_mean, s_l_varience = self._compute_column_stats(
            feature_column=sepal_length
        )
        s_w_min, s_w_max, s_w_mean, s_w_varience = self._compute_column_stats(
            feature_column=sepal_width
        )
        p_l_min, p_l_max, p_l_mean, p_l_varience = self._compute_column_stats(
            feature_column=pedal_length
        )
        p_w_min, p_w_max, p_w_mean, p_w_varience = self._compute_column_stats(
            feature_column=pedal_width
        )

        # compute within-class variance per feature
        s_l_within_var = self._compute_with_class_var(feature_col="meas_1")
        s_w_within_var = self._compute_with_class_var(feature_col="meas_2")
        p_l_within_var = self._compute_with_class_var(feature_col="meas_3")
        p_w_within_var = self._compute_with_class_var(feature_col="meas_4")

        # compute between-class variance per feature
        s_l_between_var = self._compute_between_class_var(feature_col="meas_1")
        s_w_between_var = self._compute_between_class_var(feature_col="meas_2")
        p_l_between_var = self._compute_between_class_var(feature_col="meas_3")
        p_w_between_var = self._compute_between_class_var(feature_col="meas_4")

        print("=" * 60)
        print("------------- Sepal Length Stats -------------")
        print(f"Minimum: {s_l_min}")
        print(f"Maximum: {s_l_max}")
        print(f"Mean: {s_l_mean}")
        print(f"Varience: {s_l_varience}")
        print(f"Within-Class Varience: {s_l_within_var}")
        print(f"Between-Class Varience: {s_l_between_var}")
        print("\n")

        print("------------- Sepal Width  Stats -------------")
        print(f"Minimum: {s_w_min}")
        print(f"Maximum: {s_w_max}")
        print(f"Mean: {s_w_mean}")
        print(f"Varience: {s_w_varience}")
        print(f"Within-Class Varience: {s_w_within_var}")
        print(f"Between-Class Varience: {s_w_between_var}")
        print("\n")

        print("------------- Pedal Length Stats -------------")
        print(f"Minimum: {p_l_min}")
        print(f"Maximum: {p_l_max}")
        print(f"Mean: {p_l_mean}")
        print(f"Varience: {p_l_varience}")
        print(f"Within-Class Varience: {p_l_within_var}")
        print(f"Between-Class Varience: {p_l_between_var}")
        print("\n")

        print("------------- Pedal Width Stats  -------------")
        print(f"Minimum: {p_w_min}")
        print(f"Maximum: {p_w_max}")
        print(f"Mean: {p_w_mean}")
        print(f"Varience: {p_w_varience}")
        print(f"Within-Class Varience: {p_w_within_var}")
        print(f"Between-Class Varience: {p_w_between_var}")
        print("\n")


def main(args):

    path_to_xlsx = Path(args.filename)

    iris_df = load_data(path_to_data=path_to_xlsx)

    iris_classifier = IrisClassifer(iris_df)

    iris_classifier.print_data()
    iris_classifier.compute_stats()

    print("===================================================================")

    all_features = ["meas_1", "meas_2", "meas_3", "meas_4"]
    petal_features = ["meas_3", "meas_4"]

    print("\n========== Task 1.1: Setosa vs Rest (All Features) ==========")
    print("--- Online Perceptron ---")
    p1 = OnlinePerceptron(data=iris_df, pos_class="setosa", features=all_features)
    p1.fit()
    p1.accuracy()
    print("\n========== Task 1.2: Setosa vs Rest (All Features) ==========")
    print("--- Batch Perceptron ---")
    b1 = BatchPerceptron(
        data=iris_df, pos_class="setosa", learning_rate=0.1, features=all_features
    )
    b1.fit()
    b1.accuracy()

    print("\n========== Task 2.1: Setosa vs Rest (Petal Features) ==========")
    print("--- Online Perceptron ---")
    p2 = OnlinePerceptron(data=iris_df, pos_class="setosa", features=petal_features)
    p2.fit()
    p2.accuracy()
    p2.plot_decision_boundary(
        xlabel="Petal Length",
        ylabel="Petal Width",
        title="Online: Setosa vs Rest (Petal Features)",
    )
    print("\n========== Task 2.2: Setosa vs Rest (Petal Features) ==========")
    print("--- Batch Perceptron ---")
    b2 = BatchPerceptron(
        data=iris_df, pos_class="setosa", learning_rate=0.1, features=petal_features
    )
    b2.fit()
    b2.accuracy()
    b2.plot_decision_boundary(
        xlabel="Petal Length",
        ylabel="Petal Width",
        title="Batch: Setosa vs Rest (Petal Features)",
    )

    print("\n========== Task 3.1: Virgi vs Rest (All Features) ==========")
    print("--- Online Perceptron ---")
    p5 = OnlinePerceptron(data=iris_df, pos_class="virginica", features=all_features)
    p5.fit()
    p5.accuracy()
    print("\n========== Task 3.2: Virgi vs Rest (All Features) ==========")
    print("--- Batch Perceptron ---")
    b5 = BatchPerceptron(
        data=iris_df, pos_class="virginica", learning_rate=0.1, features=all_features
    )
    b5.fit()
    b5.accuracy()

    print("\n========== Task 4.1: Virgi vs Rest (Petal Features) ==========")
    print("--- Online Perceptron ---")
    p6 = OnlinePerceptron(data=iris_df, pos_class="virginica", features=petal_features)
    p6.fit()
    p6.accuracy()
    p6.plot_decision_boundary(
        xlabel="Petal Length",
        ylabel="Petal Width",
        title="Online: Virgi vs Rest (Petal Features)",
    )
    print("\n========== Task 4.2 Virgi vs Rest (Petal Features) ==========")
    print("--- Batch Perceptron ---")
    b6 = BatchPerceptron(
        data=iris_df, pos_class="virginica", learning_rate=0.1, features=petal_features
    )
    b6.fit()
    b6.accuracy()
    b6.plot_decision_boundary(
        xlabel="Petal Length",
        ylabel="Petal Width",
        title="Batch: Virgi vs Rest (Petal Features)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Iris Classifier",
        description="Compute stats on the Iris Dataset",
        epilog="I think, I am tired",
    )
    parser.add_argument(
        "--filename",
        "-f",
        required=True,
        help="Path to the Iris dataset (must be .xlsx)",
    )

    args = parser.parse_args()
    main(args)
