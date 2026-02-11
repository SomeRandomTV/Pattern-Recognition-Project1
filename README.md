# Pattern Recognition Project 1: Iris Classification

## Overview
This project implements binary and multiclass classification algorithms on the Fisher Iris dataset using Batch Perceptron and Least Squares methods.

## Dataset
- **Source**: Fisher Iris Dataset
- **Features**: 
  - `meas_1`: Sepal Length
  - `meas_2`: Sepal Width
  - `meas_3`: Petal Length
  - `meas_4`: Petal Width
- **Classes**: Setosa, Versicolor, Virginica
- **Total Samples**: 150 (50 per class)

## Project Structure

### Part 1: Statistical Analysis
Calculate the following statistics for each feature:
- Minimum
- Maximum
- Mean
- Variance
- Within-Class Variance: `sw(i) = Σ P_j * σ²_ji`
- Between-Class Variance: `sb(i) = Σ P_j * (μ_ji - μ)²`

**Goal**: Identify which features best discriminate between classes.

### Part 2: Classification Tasks

| Task | Classes | Features | Algorithms |
|------|---------|----------|------------|
| 1 | Setosa vs (Versicolor + Virginica) | All (1-4) | Batch Perceptron, LS |
| 2 | Setosa vs (Versicolor + Virginica) | 3 & 4 only | Batch Perceptron, LS |
| 3 | Virginica vs (Versicolor + Setosa) | All (1-4) | Batch Perceptron, LS |
| 4 | Virginica vs (Versicolor + Setosa) | 3 & 4 only | Batch Perceptron, LS |
| 5 | Setosa vs Versicolor vs Virginica | 3 & 4 only | Multiclass LS |

## Algorithms

### Batch Perceptron
- Iterative learning algorithm
- Updates weights based on misclassified samples
- Converges for linearly separable data

### Least Squares (LS)
- Direct computation method
- Solves for optimal weights in closed form
- Formula: `w = (X^T X)^(-1) X^T y`

## Required Outputs

For each classification task, report:
1. **Convergence**: Did the algorithm converge?
2. **Epochs**: Number of iterations (for Batch Perceptron)
3. **Weight Vector**: Learned parameters
4. **Training Errors**: Number of misclassifications
5. **Decision Boundary Plot**: For 2-feature tasks only

## Files
- `Lastname_Firstname_Project1.py` - Python implementation
- `Lastname_Firstname_Project1.pdf` - Report with results
- `fisheriris.xlsx` - Dataset

## Dependencies
```python
numpy
pandas
matplotlib
```

## Usage
```bash
python Lastname_Firstname_Project1.py
```

## Key Formulas

### Within-Class Variance
```
sw(i) = Σ(j=1 to M) P_j * σ²_ji
```
where:
- `σ²_ji` = variance of i-th feature in class j
- `P_j` = prior probability of class j

### Between-Class Variance
```
sb(i) = Σ(j=1 to M) P_j * (μ_ji - μ)²
```
where:
- `μ_ji` = mean of i-th feature in class j
- `μ` = overall mean of i-th feature
