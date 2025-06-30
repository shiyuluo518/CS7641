"""
Enhanced Unsupervised Learning Analysis - Addressing Missing Requirements
CS7641 Machine Learning

This script addresses the missing analysis requirements:
1. PCA Eigenvalue Analysis
2. ICA Kurtosis Analysis  
3. Reconstruction Error for Randomized Projections
4. Stability of Randomized Projections
5. Data Rank and Collinearity Analysis
6. Grounded Descriptions of Resulting Clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy import stats
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

class EnhancedUnsupervisedAnalysis:
    def __init__(self):
        self.cancer_data = None
        self.bankruptcy_data = None
        self.cancer_scaled = None
        self.bankruptcy_scaled = None
        self.cancer_features = None
        self.bankruptcy_features = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess both datasets"""
        print("Loading and preprocessing datasets...")
        
        # Load Cancer Dataset
        self.cancer_data = pd.read_csv("Cancer Dataset/global_cancer_patients_2015_2024.csv")
        print(f"Cancer dataset shape: {self.cancer_data.shape}")
        
        # Load Bankruptcy Dataset
        self.bankruptcy_data = pd.read_csv("Bankruptcy Dataset/company_bankruptcy_data.csv")
        print(f"Bankruptcy dataset shape: {self.bankruptcy_data.shape}")
        
        # Preprocess Cancer Dataset
        self.cancer_features = self.cancer_data.select_dtypes(include=[np.number]).drop(['Target_Severity_Score'], axis=1, errors='ignore')
        self.cancer_features = self.cancer_features.fillna(self.cancer_features.mean())
        
        # Preprocess Bankruptcy Dataset
        self.bankruptcy_features = self.bankruptcy_data.select_dtypes(include=[np.number]).drop(['Bankrupt?'], axis=1, errors='ignore')
        self.bankruptcy_features = self.bankruptcy_features.fillna(self.bankruptcy_features.mean())
        
        # Scale the features
        scaler = StandardScaler()
        self.cancer_scaled = scaler.fit_transform(self.cancer_features)
        self.bankruptcy_scaled = scaler.fit_transform(self.bankruptcy_features)
        
        print(f"Cancer features shape: {self.cancer_scaled.shape}")
        print(f"Bankruptcy features shape: {self.bankruptcy_scaled.shape}")
    
    def analyze_data_rank_and_collinearity(self):
        """
        Analyze data rank and collinearity - addressing missing requirement
        """
        print("\n=== DATA RANK AND COLLINEARITY ANALYSIS ===")
        
        analysis_results = {}
        
        for dataset_name, data, features in [('cancer', self.cancer_scaled, self.cancer_features), 
                                           ('bankruptcy', self.bankruptcy_scaled, self.bankruptcy_features)]:
            
            # Calculate rank
            rank = np.linalg.matrix_rank(data)
            full_rank = data.shape[1]
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data.T)
            
            # Find highly correlated features (|correlation| > 0.8)
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i,j]) > 0.8:
                        high_corr_pairs.append((features.columns[i], features.columns[j], corr_matrix[i,j]))
            
            # Calculate condition number
            condition_number = np.linalg.cond(data)
            
            # Calculate variance inflation factors (VIF approximation)
            vif_scores = []
            for i in range(data.shape[1]):
                # Regress feature i on all other features
                X = np.delete(data, i, axis=1)
                y = data[:, i]
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_pred = X @ beta
                    r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                    vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
                    vif_scores.append(vif)
                except:
                    vif_scores.append(np.inf)
            
            analysis_results[dataset_name] = {
                'rank': rank,
                'full_rank': full_rank,
                'rank_deficiency': full_rank - rank,
                'condition_number': condition_number,
                'high_correlation_pairs': high_corr_pairs,
                'vif_scores': dict(zip(features.columns, vif_scores)),
                'mean_vif': np.mean([v for v in vif_scores if v != np.inf]),
                'max_vif': max([v for v in vif_scores if v != np.inf]),
                'correlation_matrix': corr_matrix
            }
            
            print(f"\n{dataset_name.upper()} DATASET:")
            print(f"  Rank: {rank}/{full_rank} ({rank/full_rank*100:.1f}% of full rank)")
            print(f"  Rank deficiency: {full_rank - rank}")
            print(f"  Condition number: {condition_number:.2e}")
            print(f"  High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")
            print(f"  Mean VIF: {analysis_results[dataset_name]['mean_vif']:.2f}")
            print(f"  Max VIF: {analysis_results[dataset_name]['max_vif']:.2f}")
            
            if high_corr_pairs:
                print("  Highly correlated feature pairs:")
                for feat1, feat2, corr in high_corr_pairs[:5]:  # Show first 5
                    print(f"    {feat1} - {feat2}: {corr:.3f}")
        
        self.results['rank_collinearity'] = analysis_results
        return analysis_results
    
    def analyze_pca_eigenvalues(self):
        """
        Analyze PCA eigenvalues distribution - addressing missing requirement
        """
        print("\n=== PCA EIGENVALUE ANALYSIS ===")
        
        pca_analysis = {}
        
        for dataset_name, data, features in [('cancer', self.cancer_scaled, self.cancer_features), 
                                           ('bankruptcy', self.bankruptcy_scaled, self.bankruptcy_features)]:
            
            # Perform full PCA to get all eigenvalues
            pca_full = PCA(random_state=42)
            pca_full.fit(data)
            
            # Get eigenvalues (squared singular values)
            eigenvalues = pca_full.singular_values_**2
            explained_variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Calculate eigenvalue statistics
            eigenvalue_stats = {
                'total_variance': np.sum(eigenvalues),
                'eigenvalues': eigenvalues,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'eigenvalue_ratio': eigenvalues[0] / eigenvalues[1] if len(eigenvalues) > 1 else np.inf,
                'condition_number': eigenvalues[0] / eigenvalues[-1] if len(eigenvalues) > 1 else 1,
                'knee_point': self._find_knee_point(cumulative_variance),
                'components_for_80_percent': np.argmax(cumulative_variance >= 0.8) + 1,
                'components_for_90_percent': np.argmax(cumulative_variance >= 0.9) + 1,
                'components_for_95_percent': np.argmax(cumulative_variance >= 0.95) + 1
            }
            
            pca_analysis[dataset_name] = eigenvalue_stats
            
            print(f"\n{dataset_name.upper()} DATASET PCA EIGENVALUES:")
            print(f"  Total variance: {eigenvalue_stats['total_variance']:.2f}")
            print(f"  Eigenvalue ratio (lambda1/lambda2): {eigenvalue_stats['eigenvalue_ratio']:.2f}")
            print(f"  Condition number: {eigenvalue_stats['condition_number']:.2e}")
            print(f"  Components for 80% variance: {eigenvalue_stats['components_for_80_percent']}")
            print(f"  Components for 90% variance: {eigenvalue_stats['components_for_90_percent']}")
            print(f"  Components for 95% variance: {eigenvalue_stats['components_for_95_percent']}")
            print(f"  Knee point: {eigenvalue_stats['knee_point']}")
            
            print("  Top 5 eigenvalues:")
            for i, (eig, var_ratio) in enumerate(zip(eigenvalues[:5], explained_variance_ratio[:5])):
                print(f"    lambda{i+1}: {eig:.3f} ({var_ratio*100:.1f}%)")
        
        self.results['pca_eigenvalues'] = pca_analysis
        return pca_analysis
    
    def _find_knee_point(self, cumulative_variance):
        """Find the knee point in cumulative variance curve"""
        # Simple method: find point where slope changes significantly
        slopes = np.diff(cumulative_variance)
        slope_changes = np.abs(np.diff(slopes))
        knee_point = np.argmax(slope_changes) + 1
        return knee_point
    
    def analyze_ica_kurtosis(self):
        """
        Analyze ICA kurtosis - addressing missing requirement
        """
        print("\n=== ICA KURTOSIS ANALYSIS ===")
        
        ica_analysis = {}
        
        for dataset_name, data, features in [('cancer', self.cancer_scaled, self.cancer_features), 
                                           ('bankruptcy', self.bankruptcy_scaled, self.bankruptcy_features)]:
            
            # Perform ICA
            ica = FastICA(n_components=min(10, data.shape[1]), random_state=42, max_iter=1000)
            ica_components = ica.fit_transform(data)
            
            # Calculate kurtosis for each component
            component_kurtosis = []
            for i in range(ica_components.shape[1]):
                kurt = kurtosis(ica_components[:, i], fisher=True)  # Fisher's definition
                component_kurtosis.append(kurt)
            
            # Calculate kurtosis statistics
            kurtosis_stats = {
                'component_kurtosis': component_kurtosis,
                'mean_kurtosis': np.mean(component_kurtosis),
                'std_kurtosis': np.std(component_kurtosis),
                'max_kurtosis': np.max(component_kurtosis),
                'min_kurtosis': np.min(component_kurtosis),
                'kurtosis_range': np.max(component_kurtosis) - np.min(component_kurtosis),
                'high_kurtosis_components': [i for i, k in enumerate(component_kurtosis) if abs(k) > 2],
                'low_kurtosis_components': [i for i, k in enumerate(component_kurtosis) if abs(k) < 0.5]
            }
            
            ica_analysis[dataset_name] = kurtosis_stats
            
            print(f"\n{dataset_name.upper()} DATASET ICA KURTOSIS:")
            print(f"  Mean kurtosis: {kurtosis_stats['mean_kurtosis']:.3f}")
            print(f"  Std kurtosis: {kurtosis_stats['std_kurtosis']:.3f}")
            print(f"  Max kurtosis: {kurtosis_stats['max_kurtosis']:.3f}")
            print(f"  Min kurtosis: {kurtosis_stats['min_kurtosis']:.3f}")
            print(f"  Kurtosis range: {kurtosis_stats['kurtosis_range']:.3f}")
            print(f"  High kurtosis components (|k| > 2): {len(kurtosis_stats['high_kurtosis_components'])}")
            print(f"  Low kurtosis components (|k| < 0.5): {len(kurtosis_stats['low_kurtosis_components'])}")
            
            print("  Component kurtosis values:")
            for i, kurt in enumerate(component_kurtosis[:5]):  # Show first 5
                print(f"    Component {i+1}: {kurt:.3f}")
        
        self.results['ica_kurtosis'] = ica_analysis
        return ica_analysis
    
    def analyze_randomized_projection_reconstruction_error(self):
        """
        Analyze reconstruction error for randomized projections - addressing missing requirement
        """
        print("\n=== RANDOMIZED PROJECTION RECONSTRUCTION ERROR ANALYSIS ===")
        
        rp_analysis = {}
        
        for dataset_name, data, features in [('cancer', self.cancer_scaled, self.cancer_features), 
                                           ('bankruptcy', self.bankruptcy_scaled, self.bankruptcy_features)]:
            
            # Perform randomized projection
            rp = GaussianRandomProjection(n_components=2, random_state=42)
            reduced_data = rp.fit_transform(data)
            
            # Attempt reconstruction (pseudo-inverse)
            try:
                # Use pseudo-inverse for reconstruction
                reconstruction_matrix = np.linalg.pinv(rp.components_)
                reconstructed_data = reduced_data @ reconstruction_matrix
                
                # Calculate reconstruction error
                mse = mean_squared_error(data, reconstructed_data)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(data - reconstructed_data))
                
                # Calculate relative error
                relative_mse = mse / np.var(data)
                relative_rmse = rmse / np.std(data)
                
                # Calculate distance preservation
                original_distances = []
                projected_distances = []
                
                # Sample some point pairs for distance comparison
                n_samples = min(1000, data.shape[0] * (data.shape[0] - 1) // 2)
                sample_pairs = []
                
                for _ in range(n_samples):
                    i, j = np.random.choice(data.shape[0], 2, replace=False)
                    sample_pairs.append((i, j))
                
                for i, j in sample_pairs:
                    orig_dist = np.linalg.norm(data[i] - data[j])
                    proj_dist = np.linalg.norm(reduced_data[i] - reduced_data[j])
                    original_distances.append(orig_dist)
                    projected_distances.append(proj_dist)
                
                # Calculate distance preservation ratio
                distance_ratios = np.array(projected_distances) / np.array(original_distances)
                distance_preservation = np.mean(distance_ratios)
                distance_preservation_std = np.std(distance_ratios)
                
                reconstruction_stats = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'relative_mse': relative_mse,
                    'relative_rmse': relative_rmse,
                    'distance_preservation': distance_preservation,
                    'distance_preservation_std': distance_preservation_std,
                    'reconstruction_matrix': reconstruction_matrix,
                    'reconstructed_data': reconstructed_data
                }
                
            except Exception as e:
                print(f"  Warning: Could not compute reconstruction for {dataset_name}: {e}")
                reconstruction_stats = {
                    'mse': np.nan,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'relative_mse': np.nan,
                    'relative_rmse': np.nan,
                    'distance_preservation': np.nan,
                    'distance_preservation_std': np.nan
                }
            
            rp_analysis[dataset_name] = reconstruction_stats
            
            print(f"\n{dataset_name.upper()} DATASET RP RECONSTRUCTION:")
            print(f"  MSE: {reconstruction_stats['mse']:.6f}")
            print(f"  RMSE: {reconstruction_stats['rmse']:.6f}")
            print(f"  MAE: {reconstruction_stats['mae']:.6f}")
            print(f"  Relative MSE: {reconstruction_stats['relative_mse']:.6f}")
            print(f"  Relative RMSE: {reconstruction_stats['relative_rmse']:.6f}")
            print(f"  Distance preservation ratio: {reconstruction_stats['distance_preservation']:.3f} ± {reconstruction_stats['distance_preservation_std']:.3f}")
        
        self.results['rp_reconstruction'] = rp_analysis
        return rp_analysis
    
    def analyze_randomized_projection_stability(self):
        """
        Analyze stability of randomized projections - addressing missing requirement
        """
        print("\n=== RANDOMIZED PROJECTION STABILITY ANALYSIS ===")
        
        stability_analysis = {}
        
        for dataset_name, data, features in [('cancer', self.cancer_scaled, self.cancer_features), 
                                           ('bankruptcy', self.bankruptcy_scaled, self.bankruptcy_features)]:
            
            n_runs = 20
            all_reduced_data = []
            all_components = []
            
            # Run multiple randomized projections
            for run in range(n_runs):
                rp = GaussianRandomProjection(n_components=2, random_state=run)
                reduced_data = rp.fit_transform(data)
                all_reduced_data.append(reduced_data)
                all_components.append(rp.components_)
            
            # Calculate stability metrics
            # 1. Variance in reduced data across runs
            reduced_data_array = np.array(all_reduced_data)
            data_variance = np.var(reduced_data_array, axis=0)
            mean_data_variance = np.mean(data_variance)
            
            # 2. Component similarity across runs
            component_similarities = []
            for i in range(n_runs):
                for j in range(i+1, n_runs):
                    # Calculate cosine similarity between components
                    comp1 = all_components[i]
                    comp2 = all_components[j]
                    
                    # Normalize components
                    comp1_norm = comp1 / np.linalg.norm(comp1, axis=1, keepdims=True)
                    comp2_norm = comp2 / np.linalg.norm(comp2, axis=1, keepdims=True)
                    
                    # Calculate similarity
                    similarity = np.abs(np.trace(comp1_norm @ comp2_norm.T)) / comp1.shape[0]
                    component_similarities.append(similarity)
            
            mean_component_similarity = np.mean(component_similarities)
            std_component_similarity = np.std(component_similarities)
            
            # 3. Distance preservation consistency
            distance_preservations = []
            for run in range(n_runs):
                reduced_data = all_reduced_data[run]
                
                # Sample point pairs
                n_samples = min(500, data.shape[0] * (data.shape[0] - 1) // 2)
                sample_pairs = []
                
                for _ in range(n_samples):
                    i, j = np.random.choice(data.shape[0], 2, replace=False)
                    sample_pairs.append((i, j))
                
                run_distances = []
                for i, j in sample_pairs:
                    orig_dist = np.linalg.norm(data[i] - data[j])
                    proj_dist = np.linalg.norm(reduced_data[i] - reduced_data[j])
                    if orig_dist > 0:
                        run_distances.append(proj_dist / orig_dist)
                
                if run_distances:
                    distance_preservations.append(np.mean(run_distances))
            
            distance_preservation_mean = np.mean(distance_preservations)
            distance_preservation_std = np.std(distance_preservations)
            
            stability_stats = {
                'n_runs': n_runs,
                'mean_data_variance': mean_data_variance,
                'mean_component_similarity': mean_component_similarity,
                'std_component_similarity': std_component_similarity,
                'distance_preservation_mean': distance_preservation_mean,
                'distance_preservation_std': distance_preservation_std,
                'stability_score': mean_component_similarity * (1 - distance_preservation_std / distance_preservation_mean) if distance_preservation_mean > 0 else 0
            }
            
            stability_analysis[dataset_name] = stability_stats
            
            print(f"\n{dataset_name.upper()} DATASET RP STABILITY ({n_runs} runs):")
            print(f"  Mean data variance: {mean_data_variance:.6f}")
            print(f"  Component similarity: {mean_component_similarity:.3f} ± {std_component_similarity:.3f}")
            print(f"  Distance preservation: {distance_preservation_mean:.3f} ± {distance_preservation_std:.3f}")
            print(f"  Stability score: {stability_stats['stability_score']:.3f}")
        
        self.results['rp_stability'] = stability_analysis
        return stability_analysis
    
    def generate_grounded_cluster_descriptions(self):
        """
        Generate grounded descriptions of resulting clusters - addressing missing requirement
        """
        print("\n=== GROUNDED CLUSTER DESCRIPTIONS ===")
        
        cluster_descriptions = {}
        
        for dataset_name, data, features, original_data in [('cancer', self.cancer_scaled, self.cancer_features, self.cancer_data), 
                                                           ('bankruptcy', self.bankruptcy_scaled, self.bankruptcy_features, self.bankruptcy_data)]:
            
            # Perform clustering
            n_clusters = 3 if dataset_name == 'cancer' else 2
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(data)
            
            # EM clustering
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            em_labels = gmm.fit_predict(data)
            
            # Analyze clusters
            kmeans_descriptions = self._analyze_cluster_characteristics(data, kmeans_labels, features, original_data, dataset_name)
            em_descriptions = self._analyze_cluster_characteristics(data, em_labels, features, original_data, dataset_name)
            
            cluster_descriptions[dataset_name] = {
                'kmeans': kmeans_descriptions,
                'em': em_descriptions
            }
            
            print(f"\n{dataset_name.upper()} DATASET CLUSTER DESCRIPTIONS:")
            
            print("\n  K-MEANS CLUSTERS:")
            for cluster_id, desc in kmeans_descriptions.items():
                print(f"    Cluster {cluster_id}: {desc['summary']}")
                print(f"      Size: {desc['size']} samples ({desc['size_percent']:.1f}%)")
                print(f"      Key characteristics: {desc['key_characteristics']}")
            
            print("\n  EM CLUSTERS:")
            for cluster_id, desc in em_descriptions.items():
                print(f"    Cluster {cluster_id}: {desc['summary']}")
                print(f"      Size: {desc['size']} samples ({desc['size_percent']:.1f}%)")
                print(f"      Key characteristics: {desc['key_characteristics']}")
        
        self.results['cluster_descriptions'] = cluster_descriptions
        return cluster_descriptions
    
    def _analyze_cluster_characteristics(self, data, labels, features, original_data, dataset_name):
        """Analyze characteristics of each cluster"""
        descriptions = {}
        n_clusters = len(np.unique(labels))
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_original = original_data.iloc[cluster_mask]
            
            # Basic statistics
            cluster_size = np.sum(cluster_mask)
            cluster_size_percent = cluster_size / len(data) * 100
            
            # Feature means for this cluster
            cluster_means = np.mean(cluster_data, axis=0)
            
            # Find most distinctive features (highest absolute z-scores)
            overall_means = np.mean(data, axis=0)
            overall_stds = np.std(data, axis=0)
            z_scores = (cluster_means - overall_means) / overall_stds
            
            # Get top 3 most distinctive features
            distinctive_features_idx = np.argsort(np.abs(z_scores))[-3:][::-1]
            distinctive_features = []
            
            for idx in distinctive_features_idx:
                feature_name = features.columns[idx]
                z_score = z_scores[idx]
                cluster_mean = cluster_means[idx]
                overall_mean = overall_means[idx]
                
                if z_score > 0.5:
                    distinctive_features.append(f"{feature_name} (high: {cluster_mean:.2f} vs {overall_mean:.2f})")
                elif z_score < -0.5:
                    distinctive_features.append(f"{feature_name} (low: {cluster_mean:.2f} vs {overall_mean:.2f})")
            
            # Generate summary description
            if dataset_name == 'cancer':
                summary = self._generate_cancer_cluster_summary(cluster_id, cluster_means, features, z_scores)
            else:
                summary = self._generate_bankruptcy_cluster_summary(cluster_id, cluster_means, features, z_scores)
            
            descriptions[cluster_id] = {
                'size': cluster_size,
                'size_percent': cluster_size_percent,
                'means': cluster_means,
                'z_scores': z_scores,
                'distinctive_features': distinctive_features,
                'key_characteristics': ', '.join(distinctive_features[:3]),
                'summary': summary
            }
        
        return descriptions
    
    def _generate_cancer_cluster_summary(self, cluster_id, cluster_means, features, z_scores):
        """Generate human-readable summary for cancer clusters"""
        # Map feature indices to names
        feature_names = list(features.columns)
        
        # Find key features
        age_idx = feature_names.index('Age') if 'Age' in feature_names else -1
        genetic_idx = feature_names.index('Genetic_Risk') if 'Genetic_Risk' in feature_names else -1
        smoking_idx = feature_names.index('Smoking') if 'Smoking' in feature_names else -1
        
        summary_parts = []
        
        if age_idx >= 0:
            age_z = z_scores[age_idx]
            if age_z > 0.5:
                summary_parts.append("older patients")
            elif age_z < -0.5:
                summary_parts.append("younger patients")
        
        if genetic_idx >= 0:
            genetic_z = z_scores[genetic_idx]
            if genetic_z > 0.5:
                summary_parts.append("high genetic risk")
            elif genetic_z < -0.5:
                summary_parts.append("low genetic risk")
        
        if smoking_idx >= 0:
            smoking_z = z_scores[smoking_idx]
            if smoking_z > 0.5:
                summary_parts.append("heavy smokers")
            elif smoking_z < -0.5:
                summary_parts.append("non-smokers")
        
        if not summary_parts:
            summary_parts.append("moderate risk profile")
        
        return f"Cancer patients with {' and '.join(summary_parts)}"
    
    def _generate_bankruptcy_cluster_summary(self, cluster_id, cluster_means, features, z_scores):
        """Generate human-readable summary for bankruptcy clusters"""
        feature_names = list(features.columns)
        
        # Find key financial ratios
        roa_idx = feature_names.index('ROA') if 'ROA' in feature_names else -1
        current_ratio_idx = feature_names.index('Current_Ratio') if 'Current_Ratio' in feature_names else -1
        debt_ratio_idx = feature_names.index('Debt_Ratio') if 'Debt_Ratio' in feature_names else -1
        
        summary_parts = []
        
        if roa_idx >= 0:
            roa_z = z_scores[roa_idx]
            if roa_z > 0.5:
                summary_parts.append("high profitability")
            elif roa_z < -0.5:
                summary_parts.append("low profitability")
        
        if current_ratio_idx >= 0:
            cr_z = z_scores[current_ratio_idx]
            if cr_z > 0.5:
                summary_parts.append("strong liquidity")
            elif cr_z < -0.5:
                summary_parts.append("poor liquidity")
        
        if debt_ratio_idx >= 0:
            debt_z = z_scores[debt_ratio_idx]
            if debt_z > 0.5:
                summary_parts.append("high leverage")
            elif debt_z < -0.5:
                summary_parts.append("low leverage")
        
        if not summary_parts:
            summary_parts.append("average financial health")
        
        return f"Companies with {' and '.join(summary_parts)}"
    
    def run_enhanced_analysis(self):
        """Run all enhanced analysis components"""
        print("Starting Enhanced Unsupervised Learning Analysis...")
        
        # Load data
        self.load_and_preprocess_data()
        
        # Run all enhanced analyses
        rank_collinearity = self.analyze_data_rank_and_collinearity()
        pca_eigenvalues = self.analyze_pca_eigenvalues()
        ica_kurtosis = self.analyze_ica_kurtosis()
        rp_reconstruction = self.analyze_randomized_projection_reconstruction_error()
        rp_stability = self.analyze_randomized_projection_stability()
        cluster_descriptions = self.generate_grounded_cluster_descriptions()
        
        # Generate comprehensive report
        self.generate_enhanced_report()
        
        return {
            'rank_collinearity': rank_collinearity,
            'pca_eigenvalues': pca_eigenvalues,
            'ica_kurtosis': ica_kurtosis,
            'rp_reconstruction': rp_reconstruction,
            'rp_stability': rp_stability,
            'cluster_descriptions': cluster_descriptions
        }
    
    def generate_enhanced_report(self):
        """Generate comprehensive enhanced analysis report"""
        print("\nGenerating enhanced analysis report...")
        
        with open('enhanced_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("ENHANCED UNSUPERVISED LEARNING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Data Rank and Collinearity
            f.write("1. DATA RANK AND COLLINEARITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for dataset_name, analysis in self.results['rank_collinearity'].items():
                f.write(f"\n{dataset_name.upper()} DATASET:\n")
                f.write(f"  Rank: {analysis['rank']}/{analysis['full_rank']} ({analysis['rank']/analysis['full_rank']*100:.1f}% of full rank)\n")
                f.write(f"  Rank deficiency: {analysis['rank_deficiency']}\n")
                f.write(f"  Condition number: {analysis['condition_number']:.2e}\n")
                f.write(f"  Mean VIF: {analysis['mean_vif']:.2f}\n")
                f.write(f"  Max VIF: {analysis['max_vif']:.2f}\n")
                f.write(f"  High correlation pairs: {len(analysis['high_correlation_pairs'])}\n")
                
                if analysis['high_correlation_pairs']:
                    f.write("  Highly correlated features:\n")
                    for feat1, feat2, corr in analysis['high_correlation_pairs'][:5]:
                        f.write(f"    {feat1} - {feat2}: {corr:.3f}\n")
            
            # PCA Eigenvalue Analysis
            f.write("\n\n2. PCA EIGENVALUE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for dataset_name, analysis in self.results['pca_eigenvalues'].items():
                f.write(f"\n{dataset_name.upper()} DATASET:\n")
                f.write(f"  Total variance: {analysis['total_variance']:.2f}\n")
                f.write(f"  Eigenvalue ratio (lambda1/lambda2): {analysis['eigenvalue_ratio']:.2f}\n")
                f.write(f"  Condition number: {analysis['condition_number']:.2e}\n")
                f.write(f"  Components for 80% variance: {analysis['components_for_80_percent']}\n")
                f.write(f"  Components for 90% variance: {analysis['components_for_90_percent']}\n")
                f.write(f"  Components for 95% variance: {analysis['components_for_95_percent']}\n")
                f.write(f"  Knee point: {analysis['knee_point']}\n")
                
                f.write("  Eigenvalue distribution:\n")
                for i, (eig, var_ratio) in enumerate(zip(analysis['eigenvalues'][:5], analysis['explained_variance_ratio'][:5])):
                    f.write(f"    lambda{i+1}: {eig:.3f} ({var_ratio*100:.1f}%)\n")
            
            # ICA Kurtosis Analysis
            f.write("\n\n3. ICA KURTOSIS ANALYSIS\n")
            f.write("-" * 25 + "\n")
            for dataset_name, analysis in self.results['ica_kurtosis'].items():
                f.write(f"\n{dataset_name.upper()} DATASET:\n")
                f.write(f"  Mean kurtosis: {analysis['mean_kurtosis']:.3f}\n")
                f.write(f"  Std kurtosis: {analysis['std_kurtosis']:.3f}\n")
                f.write(f"  Max kurtosis: {analysis['max_kurtosis']:.3f}\n")
                f.write(f"  Min kurtosis: {analysis['min_kurtosis']:.3f}\n")
                f.write(f"  Kurtosis range: {analysis['kurtosis_range']:.3f}\n")
                f.write(f"  High kurtosis components (|k| > 2): {len(analysis['high_kurtosis_components'])}\n")
                f.write(f"  Low kurtosis components (|k| < 0.5): {len(analysis['low_kurtosis_components'])}\n")
                
                f.write("  Component kurtosis values:\n")
                for i, kurt in enumerate(analysis['component_kurtosis'][:5]):
                    f.write(f"    Component {i+1}: {kurt:.3f}\n")
            
            # Randomized Projection Reconstruction Error
            f.write("\n\n4. RANDOMIZED PROJECTION RECONSTRUCTION ERROR\n")
            f.write("-" * 45 + "\n")
            for dataset_name, analysis in self.results['rp_reconstruction'].items():
                f.write(f"\n{dataset_name.upper()} DATASET:\n")
                f.write(f"  MSE: {analysis['mse']:.6f}\n")
                f.write(f"  RMSE: {analysis['rmse']:.6f}\n")
                f.write(f"  MAE: {analysis['mae']:.6f}\n")
                f.write(f"  Relative MSE: {analysis['relative_mse']:.6f}\n")
                f.write(f"  Relative RMSE: {analysis['relative_rmse']:.6f}\n")
                f.write(f"  Distance preservation ratio: {analysis['distance_preservation']:.3f} ± {analysis['distance_preservation_std']:.3f}\n")
            
            # Randomized Projection Stability
            f.write("\n\n5. RANDOMIZED PROJECTION STABILITY\n")
            f.write("-" * 35 + "\n")
            for dataset_name, analysis in self.results['rp_stability'].items():
                f.write(f"\n{dataset_name.upper()} DATASET:\n")
                f.write(f"  Number of runs: {analysis['n_runs']}\n")
                f.write(f"  Mean data variance: {analysis['mean_data_variance']:.6f}\n")
                f.write(f"  Component similarity: {analysis['mean_component_similarity']:.3f} ± {analysis['std_component_similarity']:.3f}\n")
                f.write(f"  Distance preservation: {analysis['distance_preservation_mean']:.3f} ± {analysis['distance_preservation_std']:.3f}\n")
                f.write(f"  Stability score: {analysis['stability_score']:.3f}\n")
            
            # Grounded Cluster Descriptions
            f.write("\n\n6. GROUNDED CLUSTER DESCRIPTIONS\n")
            f.write("-" * 35 + "\n")
            for dataset_name, analysis in self.results['cluster_descriptions'].items():
                f.write(f"\n{dataset_name.upper()} DATASET:\n")
                
                f.write("\n  K-MEANS CLUSTERS:\n")
                for cluster_id, desc in analysis['kmeans'].items():
                    f.write(f"    Cluster {cluster_id}: {desc['summary']}\n")
                    f.write(f"      Size: {desc['size']} samples ({desc['size_percent']:.1f}%)\n")
                    f.write(f"      Key characteristics: {desc['key_characteristics']}\n")
                
                f.write("\n  EM CLUSTERS:\n")
                for cluster_id, desc in analysis['em'].items():
                    f.write(f"    Cluster {cluster_id}: {desc['summary']}\n")
                    f.write(f"      Size: {desc['size']} samples ({desc['size_percent']:.1f}%)\n")
                    f.write(f"      Key characteristics: {desc['key_characteristics']}\n")
        
        print("Enhanced analysis report saved to 'enhanced_analysis_report.txt'")

if __name__ == "__main__":
    analyzer = EnhancedUnsupervisedAnalysis()
    results = analyzer.run_enhanced_analysis() 