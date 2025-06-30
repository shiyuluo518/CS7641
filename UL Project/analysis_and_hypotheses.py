"""
Unsupervised Learning Project - Analysis and Hypotheses
CS7641 Machine Learning

This script provides detailed analysis, hypotheses, and theoretical justifications
for the unsupervised learning algorithms applied to cancer and bankruptcy datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from unsupervised_learning_project import UnsupervisedLearningProject

class UnsupervisedLearningAnalysis:
    def __init__(self):
        self.project = UnsupervisedLearningProject()
        
    def develop_hypotheses(self):
        """
        Develop well-posed hypotheses based on dataset characteristics and theory
        """
        print("=== DEVELOPING HYPOTHESES ===")
        
        hypotheses = {
            "cancer_dataset": {
                "clustering": [
                    "H1: Cancer patients will form 3 distinct clusters based on severity levels (low, medium, high risk)",
                    "H2: EM will outperform K-means due to the continuous nature of cancer risk factors",
                    "H3: Age and genetic risk will be the primary factors driving cluster formation"
                ],
                "dimensionality_reduction": [
                    "H4: PCA will capture 80%+ variance with 2 components due to correlated risk factors",
                    "H5: ICA will reveal independent risk factors (genetic vs environmental)",
                    "H6: Randomized projection will preserve cluster structure despite random transformation"
                ]
            },
            "bankruptcy_dataset": {
                "clustering": [
                    "H7: Companies will cluster into 2 groups: healthy vs at-risk of bankruptcy",
                    "H8: Financial ratios will show clear separation between solvent and insolvent companies",
                    "H9: K-means will perform better than EM due to distinct financial categories"
                ],
                "dimensionality_reduction": [
                    "H10: PCA will reveal underlying financial health dimensions",
                    "H11: ICA will separate profitability from liquidity factors",
                    "H12: Dimensionality reduction will improve clustering performance by removing noise"
                ]
            },
            "combined_analysis": [
                "H13: PCA + K-means will be the best combination for both datasets",
                "H14: Non-linear methods (t-SNE) will reveal complex patterns not captured by linear methods",
                "H15: Neural networks will perform best with PCA-reduced features"
            ]
        }
        
        return hypotheses
    
    def justify_distance_measures(self):
        """
        Justify the choice of distance/similarity measures for clustering
        """
        print("\n=== JUSTIFICATION OF DISTANCE MEASURES ===")
        
        justifications = {
            "euclidean_distance": {
                "choice": "Primary distance measure for K-means",
                "justification": [
                    "Standard choice for continuous numerical data",
                    "Works well with standardized features",
                    "Computationally efficient and interpretable",
                    "Appropriate for both cancer risk factors and financial ratios"
                ]
            },
            "mahalanobis_distance": {
                "choice": "Implicitly used in Gaussian Mixture Models (EM)",
                "justification": [
                    "Accounts for feature correlations",
                    "Scale-invariant distance measure",
                    "Better for elliptical cluster shapes",
                    "Natural choice for multivariate normal distributions"
                ]
            },
            "cosine_similarity": {
                "choice": "Alternative for high-dimensional data",
                "justification": [
                    "Focuses on direction rather than magnitude",
                    "Useful when features have different scales",
                    "Robust to outliers in magnitude",
                    "Good for financial ratio analysis"
                ]
            }
        }
        
        return justifications
    
    def theoretical_foundations(self):
        """
        Provide theoretical foundations for each algorithm
        """
        print("\n=== THEORETICAL FOUNDATIONS ===")
        
        theory = {
            "kmeans": {
                "algorithm": "K-Means Clustering",
                "theory": [
                    "Minimizes within-cluster sum of squares (WCSS)",
                    "Assumes spherical clusters with equal variance",
                    "Converges to local minimum using Lloyd's algorithm",
                    "Sensitive to initial centroid placement"
                ],
                "assumptions": [
                    "Clusters are spherical and equally sized",
                    "Features are independent and equally important",
                    "Data is continuous and numerical"
                ]
            },
            "em": {
                "algorithm": "Expectation Maximization (Gaussian Mixture Model)",
                "theory": [
                    "Models data as mixture of Gaussian distributions",
                    "E-step: Compute posterior probabilities",
                    "M-step: Update parameters to maximize likelihood",
                    "Can model elliptical clusters with different sizes"
                ],
                "assumptions": [
                    "Data follows mixture of Gaussian distributions",
                    "Components are independent",
                    "Covariance matrices can be different"
                ]
            },
            "pca": {
                "algorithm": "Principal Component Analysis",
                "theory": [
                    "Finds directions of maximum variance",
                    "Orthogonal transformation to uncorrelated components",
                    "Minimizes reconstruction error",
                    "Eigenvalue decomposition of covariance matrix"
                ],
                "assumptions": [
                    "Linear relationships between features",
                    "Gaussian distribution of data",
                    "Variance captures important information"
                ]
            },
            "ica": {
                "algorithm": "Independent Component Analysis",
                "theory": [
                    "Finds statistically independent components",
                    "Assumes non-Gaussian source signals",
                    "Maximizes non-Gaussianity (kurtosis/negentropy)",
                    "Useful for blind source separation"
                ],
                "assumptions": [
                    "Source signals are statistically independent",
                    "Non-Gaussian distributions",
                    "Linear mixing model"
                ]
            },
            "randomized_projection": {
                "algorithm": "Randomized Projection",
                "theory": [
                    "Johnson-Lindenstrauss lemma guarantees distance preservation",
                    "Random projection matrix preserves distances",
                    "Computationally efficient for high dimensions",
                    "Approximate but fast dimensionality reduction"
                ],
                "assumptions": [
                    "Random projection preserves cluster structure",
                    "Sufficient dimensionality for distance preservation",
                    "Euclidean distances are meaningful"
                ]
            },
            "tsne": {
                "algorithm": "t-Distributed Stochastic Neighbor Embedding",
                "theory": [
                    "Non-linear dimensionality reduction",
                    "Preserves local structure and clusters",
                    "Uses t-distribution for low-dimensional embedding",
                    "Minimizes KL divergence between distributions"
                ],
                "assumptions": [
                    "Local structure is more important than global",
                    "Cluster preservation is desired",
                    "Non-linear relationships exist in data"
                ]
            }
        }
        
        return theory
    
    def dataset_characteristics(self):
        """
        Analyze characteristics of both datasets
        """
        print("\n=== DATASET CHARACTERISTICS ===")
        
        # Load data for analysis
        cancer_data = pd.read_csv("Cancer Dataset/global_cancer_patients_2015_2024.csv")
        bankruptcy_data = pd.read_csv("Bankruptcy Dataset/company_bankruptcy_data.csv")
        
        characteristics = {
            "cancer_dataset": {
                "shape": cancer_data.shape,
                "features": list(cancer_data.select_dtypes(include=[np.number]).columns),
                "target_distribution": cancer_data['Target_Severity_Score'].describe(),
                "key_features": {
                    "Age": "Continuous, likely normal distribution",
                    "Genetic_Risk": "Continuous, may be skewed",
                    "Air_Pollution": "Environmental factor, continuous",
                    "Alcohol_Use": "Behavioral factor, continuous",
                    "Smoking": "Behavioral factor, continuous",
                    "Obesity_Level": "Health factor, continuous"
                },
                "expected_clusters": 3,
                "reasoning": "Natural grouping into low, medium, high risk categories"
            },
            "bankruptcy_dataset": {
                "shape": bankruptcy_data.shape,
                "features": list(bankruptcy_data.select_dtypes(include=[np.number]).columns),
                "target_distribution": bankruptcy_data['Bankrupt?'].value_counts(),
                "key_features": {
                    "ROA": "Return on Assets - profitability measure",
                    "Operating_Gross_Margin": "Profitability indicator",
                    "Current_Ratio": "Liquidity measure",
                    "Debt_Ratio": "Leverage indicator",
                    "Asset_Turnover": "Efficiency measure"
                },
                "expected_clusters": 2,
                "reasoning": "Binary outcome: solvent vs bankrupt companies"
            }
        }
        
        return characteristics
    
    def expected_results_analysis(self):
        """
        Provide expected results and analysis for each experiment
        """
        print("\n=== EXPECTED RESULTS ANALYSIS ===")
        
        expected_results = {
            "clustering_on_original_data": {
                "cancer_kmeans": {
                    "expected_clusters": 3,
                    "expected_performance": "Moderate silhouette score (0.3-0.5)",
                    "reasoning": "Risk factors are continuous and may overlap"
                },
                "cancer_em": {
                    "expected_clusters": 3,
                    "expected_performance": "Better than K-means (0.4-0.6)",
                    "reasoning": "Can model overlapping distributions better"
                },
                "bankruptcy_kmeans": {
                    "expected_clusters": 2,
                    "expected_performance": "High silhouette score (0.6-0.8)",
                    "reasoning": "Clear separation between solvent and bankrupt"
                },
                "bankruptcy_em": {
                    "expected_clusters": 2,
                    "expected_performance": "Similar to K-means",
                    "reasoning": "Binary nature makes both methods effective"
                }
            },
            "dimensionality_reduction": {
                "pca": {
                    "expected_variance": "80-90% with 2 components",
                    "reasoning": "Financial and health data often have correlated features"
                },
                "ica": {
                    "expected_components": "Independent risk factors",
                    "reasoning": "Will separate genetic, environmental, and behavioral factors"
                },
                "randomized_projection": {
                    "expected_preservation": "Cluster structure preserved",
                    "reasoning": "Johnson-Lindenstrauss lemma guarantees distance preservation"
                }
            },
            "combined_performance": {
                "best_combination": "PCA + K-means",
                "reasoning": "PCA removes noise while preserving cluster structure",
                "expected_improvement": "10-20% better clustering metrics"
            },
            "neural_network": {
                "best_reduction": "PCA",
                "reasoning": "PCA provides most informative features for classification",
                "expected_accuracy": "75-85% with PCA-reduced features"
            }
        }
        
        return expected_results
    
    def run_comprehensive_analysis(self):
        """
        Run the complete analysis with all components
        """
        print("Starting Comprehensive Unsupervised Learning Analysis...")
        
        # Run the main experiments
        self.project.run_all_experiments()
        
        # Generate analysis components
        hypotheses = self.develop_hypotheses()
        justifications = self.justify_distance_measures()
        theory = self.theoretical_foundations()
        characteristics = self.dataset_characteristics()
        expected_results = self.expected_results_analysis()
        
        # Save comprehensive analysis
        with open('comprehensive_analysis.txt', 'w') as f:
            f.write("COMPREHENSIVE UNSUPERVISED LEARNING ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("HYPOTHESES:\n")
            f.write("-" * 20 + "\n")
            for category, hyps in hypotheses.items():
                f.write(f"\n{category.upper()}:\n")
                for hyp in hyps:
                    f.write(f"  {hyp}\n")
            
            f.write("\n\nDISTANCE MEASURE JUSTIFICATIONS:\n")
            f.write("-" * 35 + "\n")
            for measure, info in justifications.items():
                f.write(f"\n{measure.upper()}:\n")
                f.write(f"  Choice: {info['choice']}\n")
                f.write("  Justifications:\n")
                for just in info['justification']:
                    f.write(f"    - {just}\n")
            
            f.write("\n\nTHEORETICAL FOUNDATIONS:\n")
            f.write("-" * 25 + "\n")
            for algo, info in theory.items():
                f.write(f"\n{algo.upper()}:\n")
                f.write(f"  Algorithm: {info['algorithm']}\n")
                f.write("  Theory:\n")
                for theory_point in info['theory']:
                    f.write(f"    - {theory_point}\n")
                f.write("  Assumptions:\n")
                for assumption in info['assumptions']:
                    f.write(f"    - {assumption}\n")
        
        print("\nComprehensive analysis completed and saved to 'comprehensive_analysis.txt'")
        
        return {
            'hypotheses': hypotheses,
            'justifications': justifications,
            'theory': theory,
            'characteristics': characteristics,
            'expected_results': expected_results,
            'project_results': self.project.results
        }

if __name__ == "__main__":
    analyzer = UnsupervisedLearningAnalysis()
    results = analyzer.run_comprehensive_analysis() 