import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedLearningProject:
    def __init__(self):
        self.cancer_data = None
        self.bankruptcy_data = None
        self.cancer_scaled = None
        self.bankruptcy_scaled = None
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
        cancer_features = self.cancer_data.select_dtypes(include=[np.number]).drop(['Target_Severity_Score'], axis=1, errors='ignore')
        cancer_features = cancer_features.fillna(cancer_features.mean())
        
        # Preprocess Bankruptcy Dataset
        bankruptcy_features = self.bankruptcy_data.select_dtypes(include=[np.number]).drop(['Bankrupt?'], axis=1, errors='ignore')
        bankruptcy_features = bankruptcy_features.fillna(bankruptcy_features.mean())
        
        # Scale the features
        scaler = StandardScaler()
        self.cancer_scaled = scaler.fit_transform(cancer_features)
        self.bankruptcy_scaled = scaler.fit_transform(bankruptcy_features)
        
        print(f"Cancer features shape: {self.cancer_scaled.shape}")
        print(f"Bankruptcy features shape: {self.bankruptcy_scaled.shape}")
        
    def kmeans_clustering(self, data, dataset_name, n_clusters=3):
        """Perform K-Means clustering"""
        print(f"Performing K-Means clustering on {dataset_name}...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
        
        results = {
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies
        }
        
        self.results[f'kmeans_{dataset_name}'] = results
        return results
    
    def expectation_maximization(self, data, dataset_name, n_components=3):
        """Perform Expectation Maximization (Gaussian Mixture Model)"""
        print(f"Performing EM clustering on {dataset_name}...")
        
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(data)
        
        # Calculate metrics
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
        
        results = {
            'labels': labels,
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'aic': gmm.aic(data),
            'bic': gmm.bic(data),
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies
        }
        
        self.results[f'em_{dataset_name}'] = results
        return results
    
    def pca_reduction(self, data, dataset_name, n_components=2):
        """Perform Principal Component Analysis"""
        print(f"Performing PCA on {dataset_name}...")
        
        pca = PCA(n_components=n_components, random_state=42)
        reduced_data = pca.fit_transform(data)
        
        results = {
            'reduced_data': reduced_data,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'singular_values': pca.singular_values_
        }
        
        self.results[f'pca_{dataset_name}'] = results
        return results
    
    def ica_reduction(self, data, dataset_name, n_components=2):
        """Perform Independent Component Analysis"""
        print(f"Performing ICA on {dataset_name}...")
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        reduced_data = ica.fit_transform(data)
        
        results = {
            'reduced_data': reduced_data,
            'mixing_matrix': ica.mixing_,
            'unmixing_matrix': ica.components_
        }
        
        self.results[f'ica_{dataset_name}'] = results
        return results
    
    def randomized_projection(self, data, dataset_name, n_components=2):
        """Perform Randomized Projection"""
        print(f"Performing Randomized Projection on {dataset_name}...")
        
        rp = GaussianRandomProjection(n_components=n_components, random_state=42)
        reduced_data = rp.fit_transform(data)
        
        results = {
            'reduced_data': reduced_data,
            'components': rp.components_
        }
        
        self.results[f'rp_{dataset_name}'] = results
        return results
    
    def tsne_reduction(self, data, dataset_name, n_components=2):
        """Perform t-SNE (Non-linear manifold learning for extra credit)"""
        print(f"Performing t-SNE on {dataset_name}...")
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced_data = tsne.fit_transform(data)
        
        results = {
            'reduced_data': reduced_data
        }
        
        self.results[f'tsne_{dataset_name}'] = results
        return results
    
    def run_clustering_experiments(self):
        """Run clustering algorithms on original datasets"""
        print("\n=== STEP 1: Clustering on Original Datasets ===")
        
        # K-Means on Cancer Dataset
        self.kmeans_clustering(self.cancer_scaled, 'cancer', n_clusters=3)
        
        # K-Means on Bankruptcy Dataset
        self.kmeans_clustering(self.bankruptcy_scaled, 'bankruptcy', n_clusters=2)
        
        # EM on Cancer Dataset
        self.expectation_maximization(self.cancer_scaled, 'cancer', n_components=3)
        
        # EM on Bankruptcy Dataset
        self.expectation_maximization(self.bankruptcy_scaled, 'bankruptcy', n_components=2)
    
    def run_dimensionality_reduction_experiments(self):
        """Run dimensionality reduction algorithms on datasets"""
        print("\n=== STEP 2: Dimensionality Reduction ===")
        
        # PCA
        self.pca_reduction(self.cancer_scaled, 'cancer', n_components=2)
        self.pca_reduction(self.bankruptcy_scaled, 'bankruptcy', n_components=2)
        
        # ICA
        self.ica_reduction(self.cancer_scaled, 'cancer', n_components=2)
        self.ica_reduction(self.bankruptcy_scaled, 'bankruptcy', n_components=2)
        
        # Randomized Projection
        self.randomized_projection(self.cancer_scaled, 'cancer', n_components=2)
        self.randomized_projection(self.bankruptcy_scaled, 'bankruptcy', n_components=2)
        
        # t-SNE (Extra Credit)
        self.tsne_reduction(self.cancer_scaled, 'cancer', n_components=2)
        self.tsne_reduction(self.bankruptcy_scaled, 'bankruptcy', n_components=2)
    
    def run_combined_experiments(self):
        """Run clustering on dimensionality-reduced datasets"""
        print("\n=== STEP 3: Clustering on Dimensionality-Reduced Datasets ===")
        
        reduction_methods = ['pca', 'ica', 'rp', 'tsne']
        clustering_methods = ['kmeans', 'em']
        
        for reduction in reduction_methods:
            for clustering in clustering_methods:
                for dataset in ['cancer', 'bankruptcy']:
                    if f'{reduction}_{dataset}' in self.results:
                        reduced_data = self.results[f'{reduction}_{dataset}']['reduced_data']
                        
                        if clustering == 'kmeans':
                            n_clusters = 3 if dataset == 'cancer' else 2
                            self.kmeans_clustering(reduced_data, f'{dataset}_{reduction}', n_clusters)
                        else:
                            n_components = 3 if dataset == 'cancer' else 2
                            self.expectation_maximization(reduced_data, f'{dataset}_{reduction}', n_components)
    
    def neural_network_experiment(self):
        """Run neural network on dimensionality-reduced datasets"""
        print("\n=== STEP 4: Neural Network on Dimensionality-Reduced Datasets ===")
        
        # Use Cancer dataset for neural network experiment
        cancer_target = self.cancer_data['Target_Severity_Score'].values
        cancer_target_binary = (cancer_target > cancer_target.mean()).astype(int)
        
        reduction_methods = ['pca', 'ica', 'rp', 'tsne']
        
        for reduction in reduction_methods:
            if f'{reduction}_cancer' in self.results:
                reduced_data = self.results[f'{reduction}_cancer']['reduced_data']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    reduced_data, cancer_target_binary, test_size=0.2, random_state=42
                )
                
                # Train neural network
                nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                nn.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = nn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.results[f'nn_{reduction}_cancer'] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                
                print(f"Neural Network with {reduction.upper()} - Accuracy: {accuracy:.4f}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== Creating Visualizations ===")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Unsupervised Learning Results', fontsize=16)
        
        # Original data clustering results
        datasets = ['cancer', 'bankruptcy']
        clustering_methods = ['kmeans', 'em']
        
        for i, dataset in enumerate(datasets):
            for j, method in enumerate(clustering_methods):
                if f'{method}_{dataset}' in self.results:
                    result = self.results[f'{method}_{dataset}']
                    labels = result['labels']
                    
                    # Use PCA for visualization if data is high-dimensional
                    if dataset == 'cancer':
                        data_for_viz = self.cancer_scaled
                    else:
                        data_for_viz = self.bankruptcy_scaled
                    
                    # Reduce to 2D for visualization
                    pca_viz = PCA(n_components=2)
                    data_2d = pca_viz.fit_transform(data_for_viz)
                    
                    scatter = axes[i, j].scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
                    axes[i, j].set_title(f'{method.upper()} on {dataset.capitalize()}')
                    axes[i, j].set_xlabel('PC1')
                    axes[i, j].set_ylabel('PC2')
                    plt.colorbar(scatter, ax=axes[i, j])
        
        plt.tight_layout()
        plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Dimensionality reduction visualizations
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Dimensionality Reduction Results', fontsize=16)
        
        reduction_methods = ['pca', 'ica', 'rp', 'tsne']
        
        for i, dataset in enumerate(datasets):
            for j, method in enumerate(reduction_methods):
                if f'{method}_{dataset}' in self.results:
                    result = self.results[f'{method}_{dataset}']
                    reduced_data = result['reduced_data']
                    
                    axes[i, j].scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
                    axes[i, j].set_title(f'{method.upper()} on {dataset.capitalize()}')
                    axes[i, j].set_xlabel('Component 1')
                    axes[i, j].set_ylabel('Component 2')
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive results report"""
        print("\n=== Generating Results Report ===")
        
        # Create results summary
        report = []
        
        # Clustering results on original data
        print("\n1. Clustering Results on Original Data:")
        print("-" * 50)
        
        for dataset in ['cancer', 'bankruptcy']:
            for method in ['kmeans', 'em']:
                if f'{method}_{dataset}' in self.results:
                    result = self.results[f'{method}_{dataset}']
                    print(f"{method.upper()} on {dataset.capitalize()}:")
                    print(f"  Silhouette Score: {result['silhouette_score']:.4f}")
                    print(f"  Calinski-Harabasz Score: {result['calinski_harabasz_score']:.4f}")
                    print(f"  Davies-Bouldin Score: {result['davies_bouldin_score']:.4f}")
                    if method == 'kmeans':
                        print(f"  Inertia: {result['inertia']:.4f}")
                    else:
                        print(f"  AIC: {result['aic']:.4f}")
                        print(f"  BIC: {result['bic']:.4f}")
                    print()
        
        # Dimensionality reduction results
        print("\n2. Dimensionality Reduction Results:")
        print("-" * 50)
        
        for dataset in ['cancer', 'bankruptcy']:
            for method in ['pca', 'ica', 'rp']:
                if f'{method}_{dataset}' in self.results:
                    result = self.results[f'{method}_{dataset}']
                    if method == 'pca':
                        print(f"{method.upper()} on {dataset.capitalize()}:")
                        print(f"  Explained Variance Ratio: {result['explained_variance_ratio']}")
                        print(f"  Cumulative Variance: {result['cumulative_variance_ratio'][-1]:.4f}")
                    print()
        
        # Neural network results
        print("\n3. Neural Network Results:")
        print("-" * 50)
        
        for method in ['pca', 'ica', 'rp', 'tsne']:
            if f'nn_{method}_cancer' in self.results:
                result = self.results[f'nn_{method}_cancer']
                print(f"Neural Network with {method.upper()}: {result['accuracy']:.4f}")
        
        # Save detailed results to file
        with open('unsupervised_learning_results.txt', 'w') as f:
            f.write("Unsupervised Learning Project Results\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in self.results.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            f.write(f"  {k}: shape {v.shape}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
        
        print("\nResults saved to 'unsupervised_learning_results.txt'")
    
    def run_all_experiments(self):
        """Run all experiments in sequence"""
        print("Starting Unsupervised Learning Project...")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Run all experiments
        self.run_clustering_experiments()
        self.run_dimensionality_reduction_experiments()
        self.run_combined_experiments()
        self.neural_network_experiment()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    # Create and run the project
    project = UnsupervisedLearningProject()
    project.run_all_experiments() 