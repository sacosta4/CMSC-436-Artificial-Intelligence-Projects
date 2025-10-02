import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
import seaborn as sns
from IPython.display import clear_output
import time

class PerceptronLearning:
    """
    Enhanced Perceptron with learning-focused features for understanding
    """
    def __init__(self, learning_rate=0.01, max_iterations=5000, 
                 activation='hard', gain=1.0, error_threshold=1e-5, verbose=False):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.activation = activation
        self.gain = gain
        self.error_threshold = error_threshold
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.training_errors = []
        self.iterations_run = 0
        self.weight_history = []  # Track weights over time
        
    def _hard_activation(self, net):
        """Hard unipolar activation function"""
        return (net >= 0).astype(float)
    
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
            return np.sum(np.abs(y - output))
        else:
            return np.sum((y - output) ** 2)
    
    def fit(self, X, y, print_every=100):
        """
        Train the perceptron with detailed output
        """
        n_samples, n_features = X.shape
        
        # Initialize weights randomly between -0.5 and 0.5
        np.random.seed(42)
        self.weights = np.random.uniform(-0.5, 0.5, n_features)
        self.bias = np.random.uniform(-0.5, 0.5)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"STARTING TRAINING - {self.activation.upper()} ACTIVATION")
            print(f"{'='*80}")
            print(f"Initial weights: {self.weights}")
            print(f"Initial bias: {self.bias}")
            print(f"Learning rate (α): {self.learning_rate}")
            if self.activation == 'soft':
                print(f"Gain (λ): {self.gain}")
            print(f"Target error: {self.error_threshold}")
            print(f"Max iterations: {self.max_iterations}")
        
        self.training_errors = []
        self.weight_history = [self.weights.copy()]
        
        # Training loop
        for iteration in range(self.max_iterations):
            total_error = 0
            
            # Show detailed output for first few samples every N iterations
            show_detail = self.verbose and (iteration % print_every == 0 or iteration < 3)
            
            if show_detail and iteration > 0:
                print(f"\n--- Iteration {iteration} ---")
            
            for i in range(n_samples):
                # Calculate net input
                net = np.dot(X[i], self.weights) + self.bias
                
                # Apply activation function
                output = self._activation_function(net)
                
                # Calculate error
                error = y[i] - output
                
                # Show details for first sample
                if show_detail and i == 0:
                    print(f"\nSample {i}: {X[i]}")
                    print(f"  net = w₁·x₁ + w₂·x₂ + b")
                    print(f"      = {self.weights[0]:.4f}·{X[i,0]:.4f} + {self.weights[1]:.4f}·{X[i,1]:.4f} + {self.bias:.4f}")
                    print(f"      = {net:.4f}")
                    print(f"  output = activation({net:.4f}) = {output:.4f}")
                    print(f"  target = {y[i]}")
                    print(f"  error = {error:.4f}")
                    print(f"  Weight update: Δw = α · error · input")
                    print(f"                     = {self.learning_rate} · {error:.4f} · {X[i]}")
                
                # Store old weights for display
                old_weights = self.weights.copy()
                old_bias = self.bias
                
                # Update weights and bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                if show_detail and i == 0:
                    print(f"  Old weights: [{old_weights[0]:.4f}, {old_weights[1]:.4f}], bias: {old_bias:.4f}")
                    print(f"  New weights: [{self.weights[0]:.4f}, {self.weights[1]:.4f}], bias: {self.bias:.4f}")
                
                total_error += abs(error) if self.activation == 'hard' else error**2
            
            self.training_errors.append(total_error)
            self.weight_history.append(self.weights.copy())
            self.iterations_run = iteration + 1
            
            if show_detail:
                print(f"\nTotal error after iteration {iteration}: {total_error:.6f}")
                print(f"Current decision boundary: {self.weights[1]:.4f}·x₂ + {self.weights[0]:.4f}·x₁ + {self.bias:.4f} = 0")
            
            # Check stopping criterion
            if total_error < self.error_threshold:
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"✓ CONVERGED at iteration {iteration + 1}")
                    print(f"  Final error: {total_error:.6f}")
                    print(f"  Final weights: {self.weights}")
                    print(f"  Final bias: {self.bias}")
                    print(f"{'='*80}")
                else:
                    print(f"Converged at iteration {iteration + 1} with error {total_error:.6f}")
                break
        
        if self.iterations_run == self.max_iterations:
            print(f"\n⚠ Reached max iterations ({self.max_iterations}). Final error: {total_error:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data"""
        net = self._calculate_net(X)
        output = self._activation_function(net)
        
        if self.activation == 'soft':
            return (output >= 0.5).astype(int)
        return output.astype(int)
    
    def get_decision_boundary(self):
        """Return decision boundary parameters"""
        return self.weights, self.bias


def demonstrate_activation_functions():
    """
    Educational visualization of activation functions
    """
    print("\n" + "="*80)
    print("CONCEPT: ACTIVATION FUNCTIONS")
    print("="*80)
    print("\nActivation functions determine how the neuron responds to its input.")
    print("Think of it as the neuron's 'personality' - how sharply or smoothly it makes decisions.\n")
    
    net_values = np.linspace(-5, 5, 200)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Hard activation
    hard_output = (net_values >= 0).astype(float)
    ax1.plot(net_values, hard_output, 'b-', linewidth=3, label='Hard Activation')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Net Input (Σw·x + b)', fontsize=12)
    ax1.set_ylabel('Output', fontsize=12)
    ax1.set_title('Hard Unipolar Activation\n(Binary: 0 or 1)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.text(0.5, 0.8, 'Output = 1 if net ≥ 0\nOutput = 0 if net < 0', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # Soft activation with different gains
    colors = ['red', 'green', 'purple']
    for idx, gain in enumerate([0.5, 1.0, 2.0]):
        soft_output = 1 / (1 + np.exp(-gain * net_values))
        ax2.plot(net_values, soft_output, color=colors[idx], linewidth=2.5, 
                label=f'Gain λ={gain}')
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Decision threshold')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Net Input (Σw·x + b)', fontsize=12)
    ax2.set_ylabel('Output', fontsize=12)
    ax2.set_title('Soft Unipolar Activation (Sigmoid)\n(Smooth: 0 to 1)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.text(0.5, 0.8, 'Output = 1/(1 + e^(-λ·net))\nHigher gain = sharper transition', 
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    print("\nKEY INSIGHTS:")
    print("  • Hard activation: Makes instant binary decisions (like a switch)")
    print("  • Soft activation: Makes gradual probabilistic decisions (like a dimmer)")
    print("  • Higher gain: Makes soft activation behave more like hard activation")
    print("  • Lower gain: Smoother gradients, potentially more stable learning")


def demonstrate_learning_rate_effect(X, y, dataset_name='Sample'):
    """
    Show how learning rate affects convergence
    """
    print("\n" + "="*80)
    print("CONCEPT: LEARNING RATE (α)")
    print("="*80)
    print("\nLearning rate controls how big steps the perceptron takes when adjusting weights.")
    print("It's like choosing between baby steps (safe but slow) or giant leaps (fast but risky).\n")
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        print(f"Testing learning rate α = {lr}...")
        p = PerceptronLearning(learning_rate=lr, activation='hard', 
                              error_threshold=1e-5, max_iterations=1000, verbose=False)
        p.fit(X, y)
        
        ax = axes[idx]
        ax.plot(p.training_errors, linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Total Error', fontsize=11)
        ax.set_title(f'α = {lr}\nConverged: {p.iterations_run < 1000} '
                    f'(in {p.iterations_run} iterations)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add annotation
        final_error = p.training_errors[-1]
        ax.annotate(f'Final error: {final_error:.6f}', 
                   xy=(len(p.training_errors)-1, final_error),
                   xytext=(len(p.training_errors)*0.6, final_error*10),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    print("\nKEY INSIGHTS:")
    print("  • Too small (0.001): Very slow convergence, many iterations needed")
    print("  • Just right (0.01-0.1): Good balance of speed and stability")
    print("  • Too large (0.5+): May oscillate or overshoot the optimal solution")
    print("  • Rule of thumb: Start with 0.01-0.1 and adjust based on convergence")


def plot_training_progress(perceptron, X_train, y_train, dataset_name, activation):
    """
    Visualize the learning process
    """
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Error convergence
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(perceptron.training_errors, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Error', fontsize=12)
    ax1.set_title(f'Training Error Convergence\nDataset {dataset_name} - {activation.capitalize()}', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add milestone markers
    milestones = [0, len(perceptron.training_errors)//4, 
                  len(perceptron.training_errors)//2, 
                  3*len(perceptron.training_errors)//4,
                  len(perceptron.training_errors)-1]
    for m in milestones:
        if m < len(perceptron.training_errors):
            ax1.plot(m, perceptron.training_errors[m], 'ro', markersize=8)
            ax1.annotate(f'{perceptron.training_errors[m]:.2f}', 
                        xy=(m, perceptron.training_errors[m]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot 2: Weight trajectory
    ax2 = plt.subplot(1, 3, 2)
    weight_history = np.array(perceptron.weight_history)
    ax2.plot(weight_history[:, 0], label='Weight 1 (price)', linewidth=2)
    ax2.plot(weight_history[:, 1], label='Weight 2 (weight)', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Weight Value', fontsize=12)
    ax2.set_title('Weight Evolution During Training', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Decision boundary evolution
    ax3 = plt.subplot(1, 3, 3)
    colors = ['blue', 'red']
    labels = ['Small Car (0)', 'Big Car (1)']
    for class_value in [0, 1]:
        mask = y_train == class_value
        ax3.scatter(X_train[mask, 0], X_train[mask, 1], 
                   c=colors[class_value], label=labels[class_value],
                   alpha=0.5, s=20)
    
    # Show multiple decision boundaries
    x1_line = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    
    # Initial boundary
    if len(perceptron.weight_history) > 0:
        w_init = perceptron.weight_history[0]
        if abs(w_init[1]) > 1e-10:
            x2_init = -(w_init[0] * x1_line + perceptron.bias) / w_init[1]
            ax3.plot(x1_line, x2_init, 'g--', linewidth=1.5, alpha=0.5, label='Initial boundary')
    
    # Final boundary
    w_final = perceptron.weights
    if abs(w_final[1]) > 1e-10:
        x2_final = -(w_final[0] * x1_line + perceptron.bias) / w_final[1]
        ax3.plot(x1_line, x2_final, 'k-', linewidth=3, label='Final boundary')
    
    ax3.set_xlabel('Normalized Price', fontsize=12)
    ax3.set_ylabel('Normalized Weight', fontsize=12)
    ax3.set_title('Decision Boundary Movement', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_decision_boundary_explained(X, y, perceptron, dataset_name, activation):
    """
    Plot with mathematical annotations
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot points
    colors = ['blue', 'red']
    labels = ['Small Car (0)', 'Big Car (1)']
    for class_val in [0, 1]:
        mask = y == class_val
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[class_val], 
                  label=labels[class_val], alpha=0.6, s=50)
    
    # Decision boundary
    w1, w2 = perceptron.weights
    b = perceptron.bias
    x1 = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    
    if abs(w2) > 1e-10:
        x2 = -(w1 * x1 + b) / w2
        ax.plot(x1, x2, 'k-', linewidth=3, label='Decision Boundary')
        
        # Shade regions
        x2_upper = np.full_like(x1, X[:, 1].max() + 0.1)
        x2_lower = np.full_like(x1, X[:, 1].min() - 0.1)
        
        ax.fill_between(x1, x2, x2_upper, where=(x2 <= x2_upper), 
                       alpha=0.1, color='red', label='Predicted: Big Car')
        ax.fill_between(x1, x2_lower, x2, where=(x2 >= x2_lower), 
                       alpha=0.1, color='blue', label='Predicted: Small Car')
    
    # Mathematical annotations
    equation_text = f'Decision Boundary Equation:\n'
    equation_text += f'{w2:.4f}·x₂ + {w1:.4f}·x₁ + {b:.4f} = 0\n\n'
    equation_text += f'Solving for x₂:\n'
    equation_text += f'x₂ = -({w1:.4f}·x₁ + {b:.4f}) / {w2:.4f}'
    
    ax.text(0.02, 0.98, equation_text, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           fontsize=11, verticalalignment='top', family='monospace')
    
    # Classification rule
    rule_text = f'Classification Rule:\n'
    rule_text += f'If {w2:.4f}·x₂ + {w1:.4f}·x₁ + {b:.4f} ≥ 0 → Big Car\n'
    rule_text += f'If {w2:.4f}·x₂ + {w1:.4f}·x₁ + {b:.4f} < 0 → Small Car'
    
    ax.text(0.02, 0.35, rule_text, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
           fontsize=10, verticalalignment='top', family='monospace')
    
    # Weight vector visualization
    center_x, center_y = 0.5, 0.5
    scale = 0.15
    ax.arrow(center_x, center_y, w1*scale, w2*scale,
            head_width=0.03, head_length=0.03, fc='darkgreen', ec='darkgreen', linewidth=2)
    ax.text(center_x + w1*scale + 0.05, center_y + w2*scale + 0.05,
           'Weight Vector\n(perpendicular to\ndecision boundary)', 
           fontsize=10, color='darkgreen', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlabel('Normalized Price (x₁)', fontsize=13)
    ax.set_ylabel('Normalized Weight (x₂)', fontsize=13)
    ax.set_title(f'Dataset {dataset_name} - {activation.capitalize()} Activation\n'
                f'Mathematical Explanation of Decision Boundary', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    ax.set_ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    
    plt.tight_layout()
    return fig


def create_comparison_dashboard(results):
    """
    Comprehensive comparison of all experiments
    """
    fig = plt.figure(figsize=(20, 12))
    
    datasets = ['A', 'B', 'C']
    configs = [
        ('hard', '75-25'), ('hard', '25-75'),
        ('soft', '75-25'), ('soft', '25-75')
    ]
    
    for row, dataset in enumerate(datasets):
        for col, (activation, split) in enumerate(configs):
            ax = plt.subplot(3, 4, row * 4 + col + 1)
            
            metrics = results[dataset][activation][split]['test_metrics']
            
            # Metrics to display
            metric_names = ['Accuracy', 'TPR\n(Sensitivity)', 'TNR\n(Specificity)', 'FPR']
            metric_keys = ['Accuracy', 'TPR (Sensitivity)', 'TNR (Specificity)', 'FPR']
            values = [metrics[k] for k in metric_keys]
            colors_bars = ['green', 'blue', 'orange', 'red']
            
            bars = ax.bar(range(len(values)), values, color=colors_bars, alpha=0.7)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Rate', fontsize=10)
            ax.set_title(f'{dataset} | {activation} | {split}', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(metric_names)))
            ax.set_xticklabels(metric_names, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Performance Comparison Across All Experiments', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


# Main execution with learning focus
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
    
    X = df[['price', 'weight']].values
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = df['type'].values
    return X_normalized, y


def confusion_matrix_metrics(y_true, y_pred):
    """Calculate confusion matrix and all metrics"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    total = len(y_true)
    accuracy = (TP + TN) / total
    error_rate = 1 - accuracy
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy, 'Error Rate': error_rate,
        'TPR (Sensitivity)': TPR, 'TNR (Specificity)': TNR,
        'FPR': FPR, 'FNR': FNR
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PERCEPTRON LEARNING TUTORIAL")
    print("Enhanced version with educational visualizations")
    print("="*80)
    
    # Step 1: Understand activation functions
    print("\nStep 1: Understanding Activation Functions")
    input("Press Enter to see activation function comparison...")
    demonstrate_activation_functions()
    
    # Step 2: Load sample data and show learning rate effect
    print("\nStep 2: Understanding Learning Rate")
    print("We'll use Dataset A as an example...")
    
    try:
        X_sample, y_sample = load_and_normalize_data('groupA.txt')
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_sample, y_sample, train_size=0.75, stratify=y_sample, random_state=42
        )
        
        input("Press Enter to see how learning rate affects convergence...")
        demonstrate_learning_rate_effect(X_train_sample, y_train_sample, 'A')
        
        # Step 3: Detailed training example
        print("\nStep 3: Watching the Perceptron Learn (Verbose Mode)")
        print("We'll train with detailed output to see what happens at each step...")
        input("Press Enter to start training...")
        
        perceptron_demo = PerceptronLearning(
            learning_rate=0.01,
            activation='hard',
            error_threshold=1e-5,
            max_iterations=5000,
            verbose=True
        )
        perceptron_demo.fit(X_train_sample, y_train_sample, print_every=50)
        
        # Show training progress visualization
        input("\nPress Enter to see training progress visualization...")
        fig_progress = plot_training_progress(perceptron_demo, X_train_sample, 
                                             y_train_sample, 'A', 'hard')
        plt.show()
        
        # Show explained decision boundary
        input("\nPress Enter to see explained decision boundary...")
        fig_explained = plot_decision_boundary_explained(X_train_sample, y_train_sample,
                                                        perceptron_demo, 'A', 'hard')
        plt.show()
        
        print("\n" + "="*80)
        print("TUTORIAL COMPLETE!")
        print("="*80)
        print("\nYou've now seen:")
        print("  ✓ How activation functions work")
        print("  ✓ How learning rate affects convergence")
        print("  ✓ Step-by-step weight updates during training")
        print("  ✓ How the decision boundary evolves")
        print("  ✓ Mathematical interpretation of results")
        print("\nYou're ready to run the full experiments for your project!")
        
    except FileNotFoundError:
        print("\n⚠ Dataset files not found. Please ensure dataset_A.csv exists.")
        print("The tutorial shows the concepts - apply these to your actual data!")
