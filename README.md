# Pattern Recognition Project 1: Iris Classification

## Overview
This project implements one-vs-rest binary classification on the Fisher Iris dataset using Online Perceptron and Batch Perceptron algorithms. It also computes per-feature statistical analysis including within-class and between-class variance.

## Dataset
- **Source**: Fisher Iris Dataset (`Proj1DataSet.xlsx`)
- **Features**:
  - `meas_1`: Sepal Length
  - `meas_2`: Sepal Width
  - `meas_3`: Petal Length
  - `meas_4`: Petal Width
- **Classes**: Setosa, Versicolor, Virginica
- **Total Samples**: 150 (50 per class)

## Project Structure

```
├── main.py                 # Entry point — stats + all classification tasks
├── online_perceptron.py    # Online (sequential) perceptron implementation
├── batch_perceptron.py     # Batch perceptron implementation
├── eda.ipynb               # Exploratory data analysis notebook
├── Proj1DataSet.xlsx       # Iris dataset
└── README.md
```

### Part 1: Statistical Analysis
For each of the 4 features, the program computes:
- Minimum, Maximum, Mean
- Total Variance
- Within-Class Variance: `sw(i) = Σ P_j · σ²_ji`
- Between-Class Variance: `sb(i) = Σ P_j · (μ_ji − μ)²`

**Goal**: Identify which features best discriminate between classes.

### Part 2: Classification Tasks

Each task is run with **both** Online Perceptron and Batch Perceptron:

| Task | Positive Class | Features | Decision Boundary Plot |
|------|---------------|----------|----------------------|
| 1 | Setosa vs Rest | All (1–4) | No (4D) |
| 2 | Setosa vs Rest | Petal (3 & 4) | Yes |
| 3 | Virginica vs Rest | All (1–4) | No (4D) |
| 4 | Virginica vs Rest | Petal (3 & 4) | Yes |

For each task, the program reports:
1. **Convergence** — whether the algorithm converged and after how many epochs
2. **Accuracy** — correct predictions / total samples
3. **Decision Boundary Plot** — for 2-feature tasks only (petal length vs petal width)

## Algorithms

### Online Perceptron
- Updates weights **after each sample** (sequential/stochastic)
- Step activation function (threshold at 0)
- Learning rate: `η = 0.1` (default)
- Max epochs: 100

### Batch Perceptron
- Accumulates weight deltas across **all samples**, then updates once per epoch
- Same activation function and convergence criteria
- Learning rate: `η = 0.1` (default)
- Max epochs: 100

Both algorithms use the update rule `w ← w + η · error · x` (online applies per sample, batch sums then applies).

## Dependencies
```
numpy
pandas
matplotlib
openpyxl
```

## Usage
```bash
python main.py -f Proj1DataSet.xlsx
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--filename` / `-f` | Yes | Path to the Iris dataset `.xlsx` file |

## Key Formulas

### Online Perceptron Update Rule
Applied **per sample** immediately:
```
w ← w + η · (t - y) · x
b ← b + η · (t - y)
```
- `t` = true label (0 or 1)
- `y` = predicted label from activation function
- `η` = learning rate

### Batch Perceptron Update Rule
Accumulated over all samples, applied **once per epoch**:
```
Δw = Σ η · (tᵢ - yᵢ) · xᵢ   (for all misclassified samples)
Δb = Σ η · (tᵢ - yᵢ)

w ← w + Δw
b ← b + Δb
```

### Within-Class Variance
```
sw(i) = Σ(j=1 to M) P_j · σ²_ji
```
- `σ²_ji` = variance of feature i in class j
- `P_j` = prior probability of class j (N_j / N)

### Between-Class Variance
```
sb(i) = Σ(j=1 to M) P_j · (μ_ji − μ)²
```
- `μ_ji` = mean of feature i in class j
- `μ` = overall mean of feature i
