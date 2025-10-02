import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=5000, 
                 activation='hard', gain=1.0, error_threshold=1e-5):
        """
        Initialize Perceptron classifier
        
        Parameters:
        - learning_rate: alpha value for weight updates
        - max_iterations: maximum training iterations
        - activation: 'hard' or 'soft' unipolar activation
        - gain: gain parameter for soft activation
        - error_threshold: stopping criterion for total error
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.activation = activation
        self.gain = gain
        self.error_threshold = error_threshold
        self.weights = None
        self.bias = None
        self.training_errors = []
        self.iterations_run = 0
        
    def _hard_activation(self, net):
        """Hard unipolar activation function"""
        return (net >= 0).astype(int)
    
    def _soft_activation(self, net):
        """Soft unipolar (sigmoid) activation function"""
        return 1.0 / (1.0 + np.exp(-self.gain * net))
    
    def _activation_function(self, net):
        """Apply selected activation function"""
        if self.activation == 'hard':
            return self._hard_activation(net)
        else:
            return self._soft_activation(net)
    
    def _calculate_net(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights) + self.bias
    
    def _calculate_total_error(self, X, y):
        """Calculate total error for all samples"""
        net = self._calculate_net(X)
        output = self._activation_function(net)
        
        if self.activation == 'hard':
            # Sum of absolute errors
            return np.sum(np.abs(y - output))
        else:
            # Sum of squared errors
            return np.sum((y - output) ** 2)
    
    def fit(self, X, y):
        """
        Train the perceptron
        
        Parameters:
        - X: training features (n_samples, n_features)
        - y: training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights randomly between -0.5 and 0.5
        np.random.seed(42)
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        self.bias = np.random.uniform(-0.5, 0.5)
        
        self.training_errors = []
        
        # Training loop
        for iteration in range(self.max_iterations):
            total_error = 0
            
            # Loop through each training sample
            for i in range(n_samples):
                # Calculate net input
                net = np.dot(X[i], self.weights) + self.bias
                
                # Apply activation function
                output = self._activation_function(net)
                
                # Calculate error
                error = y[i] - output
                
                # Update weights and bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                total_error += abs(error) if self.activation == 'hard' else error**2
            
            self.training_errors.append(total_error)
            self.iterations_run = iteration + 1
            
            # Check stopping criterion
            if total_error < self.error_threshold:
                print(f"Converged at iteration {iteration + 1} with error {total_error:.6f}")
                break
        
        if self.iterations_run == self.max_iterations:
            print(f"Reached max iterations. Final error: {total_error:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data"""
        net = self._calculate_net(X)
        output = self._activation_function(net)
        
        # For soft activation, threshold at 0.5
        if self.activation == 'soft':
            return (output >= 0.5).astype(int)
        return output.astype(int)
    
    def get_decision_boundary(self):
        """Return decision boundary parameters for plotting"""
        # Decision boundary: w1*x1 + w2*x2 + bias = 0
        # Solve for x2: x2 = -(w1*x1 + bias) / w2
        return self.weights, self.bias


def load_and_normalize_data(filepath):
    """Load and normalize dataset from .txt or .csv file"""
    # Determine delimiter and read file
    # Try comma first, then whitespace (for .txt files)
    try:
        df = pd.read_csv(filepath, header=None, names=['price', 'weight', 'type'])
    except:
        # If comma fails, try whitespace delimiter (common in .txt files)
        df = pd.read_csv(filepath, delim_whitespace=True, header=None, 
                        names=['price', 'weight', 'type'])
    
    # Normalize features (min-max normalization)
    X = df[['price', 'weight']].values
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    y = df['type'].values
    
    return X_normalized, y


def plot_results(X_train, y_train, X_test, y_test, perceptron, 
                 dataset_name, activation, split_ratio, phase='train'):
    """Plot data points and decision boundary"""
    plt.figure(figsize=(10, 8))
    
    # Select data to plot
    if phase == 'train':
        X, y = X_train, y_train
        title_phase = 'Training'
    else:
        X, y = X_test, y_test
        title_phase = 'Testing'
    
    # Plot data points
    colors = ['blue', 'red']
    labels = ['Small Car (0)', 'Big Car (1)']
    for class_value in [0, 1]:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[class_value], label=labels[class_value],
                   alpha=0.6, s=30)
    
    # Plot decision boundary
    weights, bias = perceptron.get_decision_boundary()
    
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_line = np.linspace(x1_min, x1_max, 100)
    
    if abs(weights[1]) > 1e-10:  # Avoid division by zero
        x2_line = -(weights[0] * x1_line + bias) / weights[1]
        plt.plot(x1_line, x2_line, 'k-', linewidth=2, label='Decision Boundary')
    
    plt.xlim(x1_min, x1_max)
    plt.ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    plt.xlabel('Normalized Price', fontsize=12)
    plt.ylabel('Normalized Weight', fontsize=12)
    plt.title(f'Dataset {dataset_name} - {activation.capitalize()} Activation\n'
              f'{title_phase} Data ({split_ratio})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def confusion_matrix_metrics(y_true, y_pred):
    """Calculate confusion matrix and all metrics"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate rates
    total = len(y_true)
    accuracy = (TP + TN) / total
    error_rate = 1 - accuracy
    
    # Avoid division by zero
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
    
    metrics = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy,
        'Error Rate': error_rate,
        'TPR (Sensitivity)': TPR,
        'TNR (Specificity)': TNR,
        'FPR': FPR,
        'FNR': FNR
    }
    
    return metrics


def print_confusion_matrix(metrics):
    """Print confusion matrix in formatted table"""
    print("\nConfusion Matrix:")
    print(f"{'':20} Predicted Big (1)    Predicted Small (0)")
    print(f"Actual Big (1)       {metrics['TP']:8d}           {metrics['FN']:8d}")
    print(f"Actual Small (0)     {metrics['FP']:8d}           {metrics['TN']:8d}")
    
    print("\nMetrics:")
    print(f"Accuracy:            {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"Error Rate:          {metrics['Error Rate']:.4f} ({metrics['Error Rate']*100:.2f}%)")
    print(f"True Positive Rate:  {metrics['TPR (Sensitivity)']:.4f} ({metrics['TPR (Sensitivity)']*100:.2f}%)")
    print(f"True Negative Rate:  {metrics['TNR (Specificity)']:.4f} ({metrics['TNR (Specificity)']*100:.2f}%)")
    print(f"False Positive Rate: {metrics['FPR']:.4f} ({metrics['FPR']*100:.2f}%)")
    print(f"False Negative Rate: {metrics['FNR']:.4f} ({metrics['FNR']*100:.2f}%)")


def run_experiment(dataset_path, dataset_name, error_threshold, 
                   train_size, activation, learning_rate, gain=1.0):
    """
    Run complete experiment for one configuration
    """
    print(f"\n{'='*80}")
    print(f"Dataset {dataset_name} | Activation: {activation} | Split: {int(train_size*100)}%-{int((1-train_size)*100)}%")
    print(f"{'='*80}")
    
    # Load data
    X, y = load_and_normalize_data(dataset_path)
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=42
    )
    
    print(f"\nTraining samples: {len(y_train)} | Testing samples: {len(y_test)}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Testing class distribution: {np.bincount(y_test)}")
    
    # Train perceptron
    perceptron = Perceptron(
        learning_rate=learning_rate,
        max_iterations=5000,
        activation=activation,
        gain=gain,
        error_threshold=error_threshold
    )
    
    print(f"\nTraining with learning rate: {learning_rate}, gain: {gain}")
    perceptron.fit(X_train, y_train)
    
    print(f"\nFinal Training Error: {perceptron.training_errors[-1]:.6f}")
    print(f"Iterations: {perceptron.iterations_run}")
    print(f"Weights: {perceptron.weights}")
    print(f"Bias: {perceptron.bias}")
    
    # Make predictions
    y_train_pred = perceptron.predict(X_train)
    y_test_pred = perceptron.predict(X_test)
    
    # Calculate metrics
    print("\n--- TRAINING SET METRICS ---")
    train_metrics = confusion_matrix_metrics(y_train, y_train_pred)
    print_confusion_matrix(train_metrics)
    
    print("\n--- TESTING SET METRICS ---")
    test_metrics = confusion_matrix_metrics(y_test, y_test_pred)
    print_confusion_matrix(test_metrics)
    
    # Generate plots
    split_label = f"{int(train_size*100)}-{int((1-train_size)*100)}"
    
    fig_train = plot_results(X_train, y_train, X_test, y_test, perceptron,
                             dataset_name, activation, split_label, 'train')
    
    fig_test = plot_results(X_train, y_train, X_test, y_test, perceptron,
                            dataset_name, activation, split_label, 'test')
    
    return {
        'perceptron': perceptron,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'fig_train': fig_train,
        'fig_test': fig_test
    }


# Example usage:
if __name__ == "__main__":
    # Configuration for each dataset
    configs = {
        'A': {'path': 'groupA.txt', 'threshold': 1e-5, 'lr_hard': 0.01, 'lr_soft': 0.1},
        'B': {'path': 'groupB.txt', 'threshold': 40, 'lr_hard': 0.01, 'lr_soft': 0.1},
        'C': {'path': 'groupC.txt', 'threshold': 700, 'lr_hard': 0.01, 'lr_soft': 0.1}
    }
    
    # Run all experiments
    results = {}
    
    for dataset_name, config in configs.items():
        results[dataset_name] = {}
        
        for activation in ['hard', 'soft']:
            results[dataset_name][activation] = {}
            
            lr = config[f'lr_{activation}']
            
            for train_size in [0.75, 0.25]:
                key = f"{int(train_size*100)}-{int((1-train_size)*100)}"
                
                result = run_experiment(
                    dataset_path=config['path'],
                    dataset_name=dataset_name,
                    error_threshold=config['threshold'],
                    train_size=train_size,
                    activation=activation,
                    learning_rate=lr,
                    gain=1.0
                )
                
                results[dataset_name][activation][key] = result
                
                plt.show()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)