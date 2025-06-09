import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import joblib

# Create results directories if they don't exist
os.makedirs('cancer_results/knn', exist_ok=True)
os.makedirs('cancer_results/svm', exist_ok=True)
os.makedirs('cancer_results/nn', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Load processed data
print("Loading data...")
df = pd.read_csv('Global Cancer Patients/cancer_processed.csv')

# 2. Preprocessing
def preprocess(df):
    print("Preprocessing data...")
    df = df.copy()
    # Drop irrelevant columns
    df = df.drop(['Patient_ID', 'Target_Severity_Score'], axis=1)
    # Encode categorical variables
    for col in ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    # Features and target
    X = df.drop(['High_Severity'], axis=1)
    y = df['High_Severity']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X, y = preprocess(df)

# 3. Train/test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. k-NN: Try several k values
print("\nTraining k-NN models...")
k_values = [3, 5, 11]  # Reduced k values
results = []
for k in k_values:
    print(f"Training k-NN with k={k}")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  # Use parallel processing
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
        plt.savefig('cancer_results/knn/knn_confusion_matrix_k5.png')
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
        plt.savefig('cancer_results/knn/knn_roc_curve_k5.png')
        plt.close()
        # Classification report
        with open('cancer_results/knn/knn_classification_report_k5.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred))

# Plot accuracy vs. k
plt.figure()
plt.plot([r['k'] for r in results], [r['accuracy'] for r in results], marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy vs. k')
plt.savefig('cancer_results/knn/knn_accuracy_vs_k.png')
plt.close()

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('cancer_results/knn/knn_results.csv', index=False)

# 5. SVM with Linear and RBF kernels
def evaluate_svm(X_train, X_test, y_train, y_test, kernel, params):
    print(f"\nTraining SVM with {kernel} kernel...")
    # Grid search for hyperparameter tuning
    svm = SVC(kernel=kernel, probability=True, max_iter=1000)  # Increased max_iter
    grid_search = GridSearchCV(svm, params, cv=3, scoring='accuracy', n_jobs=-1)
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
    plt.savefig(f'cancer_results/svm/svm_{kernel}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'SVM {kernel} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'SVM ({kernel}) ROC Curve')
    plt.legend()
    plt.savefig(f'cancer_results/svm/svm_{kernel}_roc_curve.png')
    plt.close()
    
    # Save classification report
    with open(f'cancer_results/svm/svm_{kernel}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    return results

# Define parameter grids for each kernel (reduced parameter space)
linear_params = {
    'C': [1]  # Further reduced parameter space
}

rbf_params = {
    'C': [1],  # Further reduced parameter space
    'gamma': ['scale']  # Only use scale
}

# Evaluate both kernels
svm_results = []
svm_results.append(evaluate_svm(X_train, X_test, y_train, y_test, 'linear', linear_params))
svm_results.append(evaluate_svm(X_train, X_test, y_train, y_test, 'rbf', rbf_params))

# Save SVM results
df_svm_results = pd.DataFrame(svm_results)
df_svm_results.to_csv('cancer_results/svm/svm_results.csv', index=False)

# Compare kernels
plt.figure(figsize=(10, 6))
plt.bar(df_svm_results['kernel'], df_svm_results['accuracy'])
plt.title('SVM Kernel Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.savefig('cancer_results/svm/svm_kernel_comparison.png')
plt.close()

# 6. Neural Network with different activation functions
def evaluate_nn(X_train, X_test, y_train, y_test, activation, params):
    print(f"\nTraining Neural Network with {activation} activation...")
    # Grid search for hyperparameter tuning
    nn = MLPClassifier(
        activation=activation,
        max_iter=1000,  # Reduced max_iter
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5  # Reduced patience
    )
    grid_search = GridSearchCV(nn, params, cv=3, scoring='accuracy', n_jobs=-1)  # Use parallel processing
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_nn = grid_search.best_estimator_
    y_pred = best_nn.predict(X_test)
    y_proba = best_nn.predict_proba(X_test)[:,1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Save results
    results = {
        'activation': activation,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Neural Network ({activation}) Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'cancer_results/nn/nn_{activation}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'NN {activation} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Neural Network ({activation}) ROC Curve')
    plt.legend()
    plt.savefig(f'cancer_results/nn/nn_{activation}_roc_curve.png')
    plt.close()
    
    # Save classification report
    with open(f'cancer_results/nn/nn_{activation}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # Plot learning curve
    plt.figure()
    plt.plot(best_nn.loss_curve_)
    plt.title(f'Neural Network ({activation}) Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(f'cancer_results/nn/nn_{activation}_learning_curve.png')
    plt.close()
    
    return results

# Define parameter grids for each activation function (reduced parameter space)
nn_params = {
    'hidden_layer_sizes': [(50,)],  # Reduced parameter space
    'alpha': [0.01],  # Reduced parameter space
    'learning_rate_init': [0.001]
}

# Evaluate both activation functions
print("\nTraining Neural Networks...")
nn_results = []
nn_results.append(evaluate_nn(X_train, X_test, y_train, y_test, 'relu', nn_params))
nn_results.append(evaluate_nn(X_train, X_test, y_train, y_test, 'tanh', nn_params))

# Save NN results
df_nn_results = pd.DataFrame(nn_results)
df_nn_results.to_csv('cancer_results/nn/nn_results.csv', index=False)

# Compare activation functions
plt.figure(figsize=(10, 6))
plt.bar(df_nn_results['activation'], df_nn_results['accuracy'])
plt.title('Neural Network Activation Function Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.savefig('cancer_results/nn/nn_activation_comparison.png')
plt.close()

print("\nAll experiments completed!") 