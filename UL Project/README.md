# Unsupervised Learning Project - CS7641

## Project Overview

This project implements and analyzes five unsupervised learning algorithms on cancer and bankruptcy datasets:

### Clustering Algorithms
1. **K-Means Clustering** - Partition-based clustering using Euclidean distance
2. **Expectation Maximization (EM)** - Gaussian Mixture Model clustering

### Dimensionality Reduction Algorithms
3. **Principal Component Analysis (PCA)** - Linear dimensionality reduction
4. **Independent Component Analysis (ICA)** - Independent component extraction
5. **Randomized Projection (RP)** - Random projection for dimensionality reduction

### Extra Credit
6. **t-SNE** - Non-linear manifold learning for visualization

## Project Structure

```
├── unsupervised_learning_project.py    # Main implementation
├── analysis_and_hypotheses.py          # Detailed analysis and hypotheses
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── Cancer Dataset/                     # Cancer patient data
└── Bankruptcy Dataset/                 # Company bankruptcy data
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Analysis
```bash
python analysis_and_hypotheses.py
```

### Run Main Experiments Only
```bash
python unsupervised_learning_project.py
```

## Experiments Performed

### Step 1: Clustering on Original Datasets
- K-Means clustering on Cancer dataset (3 clusters)
- K-Means clustering on Bankruptcy dataset (2 clusters)
- EM clustering on Cancer dataset (3 components)
- EM clustering on Bankruptcy dataset (2 components)

### Step 2: Dimensionality Reduction
- PCA on both datasets (2 components)
- ICA on both datasets (2 components)
- Randomized Projection on both datasets (2 components)
- t-SNE on both datasets (2 components) - Extra Credit

### Step 3: Clustering on Reduced Data
- All clustering algorithms applied to all dimensionality-reduced datasets
- Results compared across 12 combinations

### Step 4: Neural Network Evaluation
- Neural network trained on dimensionality-reduced Cancer dataset
- Performance comparison across all reduction methods

## Output Files

- `clustering_results.png` - Visualization of clustering results
- `dimensionality_reduction_results.png` - Visualization of dimensionality reduction
- `unsupervised_learning_results.txt` - Detailed numerical results
- `comprehensive_analysis.txt` - Complete analysis with hypotheses and theory

## Key Features

### Distance/Similarity Measures Justified
1. **Euclidean Distance** - Primary choice for K-means
   - Standard for continuous numerical data
   - Works well with standardized features
   - Computationally efficient

2. **Mahalanobis Distance** - Implicit in EM/GMM
   - Accounts for feature correlations
   - Scale-invariant
   - Better for elliptical clusters

3. **Cosine Similarity** - Alternative for high-dimensional data
   - Focuses on direction rather than magnitude
   - Robust to outliers

### Hypotheses Developed

#### Cancer Dataset
- H1: 3 distinct clusters based on severity levels
- H2: EM will outperform K-means due to continuous risk factors
- H3: Age and genetic risk drive cluster formation
- H4: PCA will capture 80%+ variance with 2 components
- H5: ICA will reveal independent risk factors

#### Bankruptcy Dataset
- H7: 2 clusters: healthy vs at-risk companies
- H8: Clear separation between solvent and insolvent
- H9: K-means better than EM due to distinct categories
- H10: PCA reveals underlying financial health dimensions

#### Combined Analysis
- H13: PCA + K-means best combination
- H14: t-SNE reveals complex non-linear patterns
- H15: Neural networks perform best with PCA

## Theoretical Foundations

### K-Means
- Minimizes within-cluster sum of squares
- Assumes spherical clusters with equal variance
- Uses Lloyd's algorithm for convergence

### EM (Gaussian Mixture Model)
- Models data as mixture of Gaussian distributions
- E-step: Compute posterior probabilities
- M-step: Update parameters to maximize likelihood

### PCA
- Finds directions of maximum variance
- Orthogonal transformation to uncorrelated components
- Minimizes reconstruction error

### ICA
- Finds statistically independent components
- Assumes non-Gaussian source signals
- Maximizes non-Gaussianity

### Randomized Projection
- Johnson-Lindenstrauss lemma guarantees distance preservation
- Computationally efficient for high dimensions
- Approximate but fast dimensionality reduction

### t-SNE
- Non-linear dimensionality reduction
- Preserves local structure and clusters
- Uses t-distribution for low-dimensional embedding

## Results Analysis

The project provides comprehensive analysis including:
- Clustering performance metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Dimensionality reduction effectiveness
- Combined algorithm performance
- Neural network classification accuracy
- Visualizations and comparisons

## Extra Credit Implementation

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** is implemented as the non-linear manifold learning algorithm because:
- It preserves local structure and cluster relationships
- Provides excellent visualization capabilities
- Can reveal complex non-linear patterns not captured by linear methods
- Particularly effective for high-dimensional data visualization

## Contact

For questions about this implementation, please refer to the comprehensive analysis files generated by the scripts. 