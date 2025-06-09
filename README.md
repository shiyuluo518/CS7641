# Enhanced Neural Network Experiments

## Overview
This project explores the impact of weight initialization, regularization techniques, and batch normalization on neural network performance. It includes experiments on two datasets: Company Bankruptcy and Global Cancer Patients.

## Requirements
- Python 3.6+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

### 1. Data Exploration
First, run the data exploration script to understand the datasets:
```bash
python data_exploration.py
```
This will generate:
- Dataset statistics
- Feature distributions
- Correlation matrices
- Class distribution plots

### 2. Base Classification Experiments
Run the base classification experiments for both datasets:
```bash
python bankruptcy_classification.py
python cancer_classification.py
```
These will generate:
- Learning curves for k-NN, SVM, and Neural Networks
- Confusion matrices
- ROC curves
- Classification reports
- Performance comparison plots

### 3. Enhanced Neural Network Experiments
Run the enhanced neural network experiments:
```bash
python nn_enhanced.py
```
This will generate results in the `enhanced_nn_results` directory:

#### Weight Initialization Results
- `enhanced_nn_results/initialization/bankruptcy/`
  - Xavier, He, and Uniform initialization comparisons
  - Weight distribution plots
  - Learning curves
  - Performance metrics
- `enhanced_nn_results/initialization/cancer/`
  - Same structure as above for cancer dataset

#### Regularization Results
- `enhanced_nn_results/regularization/bankruptcy/`
  - L2 regularization (0.001, 0.01)
  - Dropout (0.1, 0.2)
  - Baseline (no regularization)
  - Learning curves and performance metrics
- `enhanced_nn_results/regularization/cancer/`
  - Same structure as above for cancer dataset

#### Batch Normalization Results
- `enhanced_nn_results/batch_norm/bankruptcy/`
  - With/without batch normalization
  - Momentum values (0.1, 0.9)
  - Activation distributions
  - Learning curves
- `enhanced_nn_results/batch_norm/cancer/`
  - Same structure as above for cancer dataset

## Output Structure
```
project_root/
├── data_exploration.py
├── bankruptcy_classification.py
├── cancer_classification.py
├── nn_enhanced.py
├── enhanced_nn_results/
│   ├── initialization/
│   │   ├── bankruptcy/
│   │   └── cancer/
│   ├── regularization/
│   │   ├── bankruptcy/
│   │   └── cancer/
│   └── batch_norm/
│       ├── bankruptcy/
│       └── cancer/
└── README.md
```

## Results Analysis
Each experiment generates:
- Learning curves
- Weight and activation distributions
- Confusion matrices
- ROC curves
- Classification reports
- Performance comparison plots

## Hypothesis
The central hypothesis is that the choice of weight initialization, regularization techniques, and batch normalization significantly influences neural network performance and convergence.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 