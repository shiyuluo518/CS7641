import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Create results directories
os.makedirs('bankruptcy_results/knn', exist_ok=True)
os.makedirs('bankruptcy_results/svm', exist_ok=True)
os.makedirs('bankruptcy_results/nn', exist_ok=True)

# 1. Load processed data
df = pd.read_csv('Company Bankruptcy/bankruptcy_processed.csv')

# Print dataset information
print("\nDataset Information:")
print("-------------------")
print(f"Total samples: {len(df)}")
print("\nClass Distribution:")
print(df['Bankrupt?'].value_counts())
print("\nClass Distribution (%):")
print(df['Bankrupt?'].value_counts(normalize=True) * 100)

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Bankrupt?')
plt.title('Class Distribution in Bankruptcy Dataset')
plt.xlabel('Bankruptcy Status')
plt.ylabel('Count')
plt.savefig('bankruptcy_results/class_distribution.png')
plt.close()

# Print feature information
print("\nFeature Information:")
print("-------------------")
print(f"Number of features: {len(df.columns) - 1}")  # Excluding target
print("\nFeature names:")
print(df.columns.tolist())

# 2. Preprocessing
def preprocess(df):
    df = df.copy()
    # Features and target
    X = df.drop(['Bankrupt?'], axis=1)
    y = df['Bankrupt?']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X, y = preprocess(df)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining/Test Split Information:")
print("------------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nTraining set class distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts())

# 4. k-NN: Try several k values
k_values = [1, 3, 5, 11, 21]
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'k': k, 'accuracy': acc})
    if k == 5:  # Save confusion matrix and ROC for k=5
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('k-NN Confusion Matrix (k=5)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('bankruptcy_results/knn/knn_confusion_matrix_k5.png')
        plt.close()
        # ROC curve
        y_proba = knn.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'k-NN (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('k-NN ROC Curve (k=5)')
        plt.legend()
        plt.savefig('bankruptcy_results/knn/knn_roc_curve_k5.png')
        plt.close()
        # Classification report
        with open('bankruptcy_results/knn/knn_classification_report_k5.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred))

# Plot accuracy vs. k
plt.figure()
plt.plot([r['k'] for r in results], [r['accuracy'] for r in results], marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy vs. k')
plt.savefig('bankruptcy_results/knn/knn_accuracy_vs_k.png')
plt.close()

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('bankruptcy_results/knn/knn_results.csv', index=False)

# 5. SVM with Linear and RBF kernels
def evaluate_svm(X_train, X_test, y_train, y_test, kernel, params):
    # Grid search for hyperparameter tuning
    svm = SVC(kernel=kernel, probability=True)
    grid_search = GridSearchCV(svm, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    y_proba = best_svm.predict_proba(X_test)[:,1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Save results
    results = {
        'kernel': kernel,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'SVM ({kernel}) Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'bankruptcy_results/svm/svm_{kernel}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'SVM {kernel} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'SVM ({kernel}) ROC Curve')
    plt.legend()
    plt.savefig(f'bankruptcy_results/svm/svm_{kernel}_roc_curve.png')
    plt.close()
    
    # Save classification report
    with open(f'bankruptcy_results/svm/svm_{kernel}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    return results

# Define parameter grids for each kernel
linear_params = {
    'C': [0.1, 1, 10, 100]
}

rbf_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}

# Evaluate both kernels
svm_results = []
svm_results.append(evaluate_svm(X_train, X_test, y_train, y_test, 'linear', linear_params))
svm_results.append(evaluate_svm(X_train, X_test, y_train, y_test, 'rbf', rbf_params))

# Save SVM results
df_svm_results = pd.DataFrame(svm_results)
df_svm_results.to_csv('bankruptcy_results/svm/svm_results.csv', index=False)

# Compare kernels
plt.figure(figsize=(10, 6))
plt.bar(df_svm_results['kernel'], df_svm_results['accuracy'])
plt.title('SVM Kernel Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.savefig('bankruptcy_results/svm/svm_kernel_comparison.png')
plt.close()

# 6. Enhanced Neural Network implementation
def evaluate_nn(X_train, X_test, y_train, y_test, activation):
    # Simplified feature selection - use fewer features
    k_best = SelectKBest(f_classif, k=20)  # Reduced from 30 to 20 features
    
    # Create pipeline with feature selection and SMOTE
    pipeline = Pipeline([
        ('feature_selection', k_best),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),  # Limit SMOTE to 50% of majority class
        ('classifier', MLPClassifier(
            activation=activation,
            max_iter=300,  # Reduced from 1000
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,  # Reduced from 10
            verbose=False
        ))
    ])
    
    # Simplified parameter grid
    params = {
        'classifier__hidden_layer_sizes': [(50,), (100,)],  # Removed complex architectures
        'classifier__alpha': [0.001, 0.01],  # Reduced options
        'classifier__learning_rate_init': [0.001],  # Single learning rate
        'classifier__batch_size': ['auto'],  # Single batch size
        'classifier__learning_rate': ['constant']  # Single learning rate strategy
    }
    
    # Use RandomizedSearchCV with fewer iterations
    grid_search = RandomizedSearchCV(
        pipeline,
        params,
        n_iter=4,  # Reduced from 8
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print(f"\nTraining Neural Network with {activation} activation...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_pipeline = grid_search.best_estimator_
    print("Selected features:", best_pipeline.named_steps['feature_selection'].get_support())
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:,1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Save results
    results = {
        'activation': activation,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'best_score': grid_search.best_score_,
        'n_iterations': len(best_pipeline.named_steps['classifier'].loss_curve_),
        'selected_features': best_pipeline.named_steps['feature_selection'].get_support().tolist()
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Neural Network ({activation}) Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'bankruptcy_results/nn/nn_{activation}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'NN {activation} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Neural Network ({activation}) ROC Curve')
    plt.legend()
    plt.savefig(f'bankruptcy_results/nn/nn_{activation}_roc_curve.png')
    plt.close()
    
    # Save classification report
    with open(f'bankruptcy_results/nn/nn_{activation}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # Plot learning curve
    plt.figure()
    plt.plot(best_pipeline.named_steps['classifier'].loss_curve_)
    plt.title(f'Neural Network ({activation}) Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(f'bankruptcy_results/nn/nn_{activation}_learning_curve.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'Feature': df.drop(['Bankrupt?'], axis=1).columns,
        'Selected': best_pipeline.named_steps['feature_selection'].get_support()
    })
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance[feature_importance['Selected']], 
                x='Feature', y='Selected')
    plt.xticks(rotation=90)
    plt.title(f'Selected Features for {activation} Neural Network')
    plt.tight_layout()
    plt.savefig(f'bankruptcy_results/nn/nn_{activation}_feature_importance.png')
    plt.close()
    
    return results

# Evaluate both activation functions
print("\nStarting Neural Network evaluation...")
nn_results = []
nn_results.append(evaluate_nn(X_train, X_test, y_train, y_test, 'relu'))
nn_results.append(evaluate_nn(X_train, X_test, y_train, y_test, 'tanh'))

# Save NN results
df_nn_results = pd.DataFrame(nn_results)
df_nn_results.to_csv('bankruptcy_results/nn/nn_results.csv', index=False)

# Compare activation functions
plt.figure(figsize=(10, 6))
plt.bar(df_nn_results['activation'], df_nn_results['accuracy'])
plt.title('Neural Network Activation Function Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.savefig('bankruptcy_results/nn/nn_activation_comparison.png')
plt.close() 