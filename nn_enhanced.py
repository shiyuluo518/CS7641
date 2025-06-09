import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create results directory for enhanced NN experiments
os.makedirs('enhanced_nn_results', exist_ok=True)
os.makedirs('enhanced_nn_results/initialization', exist_ok=True)
os.makedirs('enhanced_nn_results/regularization', exist_ok=True)
os.makedirs('enhanced_nn_results/batch_norm', exist_ok=True)

class SimpleNN(nn.Module):
    def __init__(self, input_size, use_dropout=False, use_bn=False, dropout_rate=0.2, bn_momentum=0.1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # Reduced from 50
        self.fc2 = nn.Linear(32, 16)  # Reduced from 25
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.bn1 = nn.BatchNorm1d(32) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(16) if use_bn else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_data(dataset_name):
    """Load and preprocess data based on dataset name."""
    if dataset_name == 'bankruptcy':
        df = pd.read_csv('Company Bankruptcy/bankruptcy_processed.csv')
        X = df.drop(['Bankrupt?'], axis=1)
        y = df['Bankrupt?']
    else:  # cancer dataset
        df = pd.read_csv('Global Cancer Patients/cancer_processed.csv')
        # Drop non-numeric columns (e.g., Patient_ID) for cancer dataset
        X = df.drop(['Patient_ID', 'High_Severity'], axis=1)
        y = df['High_Severity']
        # Encode categorical columns
        categorical_columns = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_proba, model_name, save_dir):
    """Evaluate model performance and save results."""
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_roc_curve.png')
    plt.close()
    # Save classification report
    with open(f'{save_dir}/{model_name.lower().replace(" ", "_")}_classification_report.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred))
    return accuracy, roc_auc

def train_model(model, train_loader, criterion, optimizer, max_epochs=10, patience=3):  # Reduced epochs and patience
    """Train model with early stopping."""
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 2 == 0:  # Print more frequently
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

def plot_weight_distribution(model, init_name, save_dir):
    """Plot the distribution of weights for each layer."""
    plt.figure(figsize=(15, 5))
    
    # Get all weight layers
    weight_layers = [(name, param) for name, param in model.named_parameters() if 'weight' in name]
    
    # Plot weights for each layer
    for i, (name, param) in enumerate(weight_layers):
        plt.subplot(1, len(weight_layers), i+1)
        plt.hist(param.data.numpy().flatten(), bins=50)
        plt.title(f'{name} Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{init_name.lower()}_weight_distribution.png')
    plt.close()

def experiment_weight_initialization(X_train, X_test, y_train, y_test, dataset_name):
    """Compare different weight initialization methods using PyTorch."""
    print(f"\nExperimenting with weight initialization on {dataset_name} dataset...")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    # Create data loaders with larger batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Use all three initialization methods
    init_methods = {
        'Xavier': 'xavier',
        'He': 'he',
        'Uniform': 'uniform'
    }

    results = []
    for init_name, init_method in init_methods.items():
        print(f"\nTraining with {init_name} initialization...")
        model = SimpleNN(X_train.shape[1])
        # Apply initialization
        if init_method == 'xavier':
            torch.nn.init.xavier_uniform_(model.fc1.weight)
            torch.nn.init.xavier_uniform_(model.fc2.weight)
            torch.nn.init.xavier_uniform_(model.fc3.weight)
        elif init_method == 'he':
            torch.nn.init.kaiming_uniform_(model.fc1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(model.fc2.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(model.fc3.weight, nonlinearity='relu')
        elif init_method == 'uniform':
            torch.nn.init.uniform_(model.fc1.weight)
            torch.nn.init.uniform_(model.fc2.weight)
            torch.nn.init.uniform_(model.fc3.weight)
        
        # Plot initial weight distribution
        plot_weight_distribution(model, init_name, f"enhanced_nn_results/initialization/{dataset_name}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train with early stopping
        losses = train_model(model, train_loader, criterion, optimizer)
        
        # Plot final weight distribution
        plot_weight_distribution(model, f"{init_name}_final", f"enhanced_nn_results/initialization/{dataset_name}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).argmax(dim=1).numpy()
            y_proba = torch.softmax(model(X_test_tensor), dim=1)[:,1].numpy()
        
        # Evaluate and save results
        accuracy, roc_auc = evaluate_model(
            y_test, y_pred, y_proba,
            f"{init_name} Initialization",
            f"enhanced_nn_results/initialization/{dataset_name}"
        )
        
        results.append({
            'initialization': init_name,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        })
        
        # Plot learning curve
        plt.figure()
        plt.plot(losses)
        plt.title(f'{init_name} Initialization Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'enhanced_nn_results/initialization/{dataset_name}/{init_name.lower()}_learning_curve.png')
        plt.close()
    
    # Save comparison results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'enhanced_nn_results/initialization/{dataset_name}/initialization_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['initialization'], df_results['accuracy'])
    plt.title('Weight Initialization Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(f'enhanced_nn_results/initialization/{dataset_name}/initialization_comparison.png')
    plt.close()

def experiment_regularization(X_train, X_test, y_train, y_test, dataset_name):
    """Compare different regularization techniques using PyTorch."""
    print(f"\nExperimenting with regularization on {dataset_name} dataset...")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    # Create data loaders with larger batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Use more regularization options
    reg_types = {
        'L2_0.001': {'use_dropout': False, 'weight_decay': 0.001},
        'L2_0.01': {'use_dropout': False, 'weight_decay': 0.01},
        'Dropout_0.1': {'use_dropout': True, 'dropout_rate': 0.1, 'weight_decay': 0.0},
        'Dropout_0.2': {'use_dropout': True, 'dropout_rate': 0.2, 'weight_decay': 0.0},
        'None': {'use_dropout': False, 'weight_decay': 0.0}
    }

    results = []
    for reg_name, reg_opts in reg_types.items():
        print(f"\nTraining with {reg_name} regularization...")
        model = SimpleNN(X_train.shape[1], 
                        use_dropout=reg_opts['use_dropout'],
                        dropout_rate=reg_opts.get('dropout_rate', 0.2))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=reg_opts['weight_decay'])
        
        # Train with early stopping
        losses = train_model(model, train_loader, criterion, optimizer)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).argmax(dim=1).numpy()
            y_proba = torch.softmax(model(X_test_tensor), dim=1)[:,1].numpy()
        
        # Evaluate and save results
        accuracy, roc_auc = evaluate_model(
            y_test, y_pred, y_proba,
            f"{reg_name} Regularization",
            f"enhanced_nn_results/regularization/{dataset_name}"
        )
        
        results.append({
            'regularization': reg_name,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        })
        
        # Plot learning curve
        plt.figure()
        plt.plot(losses)
        plt.title(f'{reg_name} Regularization Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'enhanced_nn_results/regularization/{dataset_name}/{reg_name.lower()}_learning_curve.png')
        plt.close()
    
    # Save comparison results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'enhanced_nn_results/regularization/{dataset_name}/regularization_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(15, 6))
    plt.bar(df_results['regularization'], df_results['accuracy'])
    plt.title('Regularization Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'enhanced_nn_results/regularization/{dataset_name}/regularization_comparison.png')
    plt.close()

def plot_activation_distribution(model, X, use_bn, save_dir):
    """Plot the distribution of activations before and after batch normalization."""
    model.eval()
    with torch.no_grad():
        # Get activations from first layer
        x = model.fc1(X)
        if use_bn:
            x_bn = model.bn1(x)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(x.numpy().flatten(), bins=50)
        plt.title('Activation Distribution Before BN')
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        
        if use_bn:
            plt.subplot(1, 2, 2)
            plt.hist(x_bn.numpy().flatten(), bins=50)
            plt.title('Activation Distribution After BN')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/activation_distribution_{"with" if use_bn else "without"}_bn.png')
        plt.close()

def experiment_batch_normalization(X_train, X_test, y_train, y_test, dataset_name):
    """Compare models with and without batch normalization."""
    print(f"\nExperimenting with batch normalization on {dataset_name} dataset...")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    # Create data loaders with larger batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Use two momentum values
    momentum_values = [0.1, 0.9]
    results = []
    
    for use_bn in [True, False]:
        if use_bn:
            for momentum in momentum_values:
                print(f"\nTraining with batch normalization (momentum={momentum})...")
                model = SimpleNN(X_train.shape[1], use_bn=True, bn_momentum=momentum)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Train with early stopping
                losses = train_model(model, train_loader, criterion, optimizer)
                
                # Plot activation distribution
                plot_activation_distribution(model, X_train_tensor, use_bn, 
                                          f"enhanced_nn_results/batch_norm/{dataset_name}")
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor).argmax(dim=1).numpy()
                    y_proba = torch.softmax(model(X_test_tensor), dim=1)[:,1].numpy()
                
                # Evaluate and save results
                accuracy, roc_auc = evaluate_model(
                    y_test, y_pred, y_proba,
                    f"Batch Norm (momentum={momentum})",
                    f"enhanced_nn_results/batch_norm/{dataset_name}"
                )
                
                results.append({
                    'batch_norm': f'With (momentum={momentum})',
                    'accuracy': accuracy,
                    'roc_auc': roc_auc
                })
                
                # Plot learning curve
                plt.figure()
                plt.plot(losses)
                plt.title(f'Batch Norm (momentum={momentum}) Learning Curve')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.savefig(f'enhanced_nn_results/batch_norm/{dataset_name}/bn_momentum_{momentum}_learning_curve.png')
                plt.close()
        else:
            print(f"\nTraining without batch normalization...")
            model = SimpleNN(X_train.shape[1], use_bn=False)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train with early stopping
            losses = train_model(model, train_loader, criterion, optimizer)
            
            # Plot activation distribution
            plot_activation_distribution(model, X_train_tensor, use_bn,
                                      f"enhanced_nn_results/batch_norm/{dataset_name}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor).argmax(dim=1).numpy()
                y_proba = torch.softmax(model(X_test_tensor), dim=1)[:,1].numpy()
            
            # Evaluate and save results
            accuracy, roc_auc = evaluate_model(
                y_test, y_pred, y_proba,
                "Without Batch Normalization",
                f"enhanced_nn_results/batch_norm/{dataset_name}"
            )
            
            results.append({
                'batch_norm': 'Without',
                'accuracy': accuracy,
                'roc_auc': roc_auc
            })
            
            # Plot learning curve
            plt.figure()
            plt.plot(losses)
            plt.title('Without Batch Normalization Learning Curve')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(f'enhanced_nn_results/batch_norm/{dataset_name}/without_bn_learning_curve.png')
            plt.close()
    
    # Save comparison results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'enhanced_nn_results/batch_norm/{dataset_name}/batch_norm_comparison.csv', index=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df_results['batch_norm'], df_results['accuracy'])
    plt.title('Batch Normalization Comparison - Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'enhanced_nn_results/batch_norm/{dataset_name}/batch_norm_comparison.png')
    plt.close()

def main():
    # Run experiments for both datasets
    datasets = ['bankruptcy', 'cancer']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Running experiments on {dataset} dataset")
        print(f"{'='*50}")
        
        # Load data
        X_train, X_test, y_train, y_test = load_data(dataset)
        
        # Run experiments
        experiment_weight_initialization(X_train, X_test, y_train, y_test, dataset)
        experiment_regularization(X_train, X_test, y_train, y_test, dataset)
        experiment_batch_normalization(X_train, X_test, y_train, y_test, dataset)

if __name__ == "__main__":
    main() 