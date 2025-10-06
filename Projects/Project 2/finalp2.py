import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os


def normalize(dataset, set_name):
    """Load and normalize dataset"""
    df = pd.read_csv(dataset, names=["price", "weight", "type"])

    # Normalizing the data using min-max normalization
    df["p_n"] = (df["price"] - df["price"].min()) / (
        df["price"].max() - df["price"].min()
    )
    df["w_n"] = (df["weight"] - df["weight"].min()) / (
        df["weight"].max() - df["weight"].min()
    )

    return df[["p_n", "w_n"]].values, df["type"].values


def soft_act_test(net):
    """Soft unipolar (sigmoid) activation function"""
    gain = 1.0
    return 1 / (1 + np.exp(-net * gain))


def hard_act_test(net):
    """Hard unipolar activation function"""
    if net >= 0:
        return 1
    else:
        return 0


def soft_activation(X, Y, test_percent, ni, terr):
    """Train perceptron with soft unipolar activation function"""
    alpha = 0.1
    epsilon = terr
    n = X.shape[1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_percent, random_state=42, stratify=Y
    )

    # Initialize the neuron with random weights between -0.5 and 0.5
    np.random.seed(42)
    weights = np.random.uniform(-0.5, 0.5, n)
    bias = np.random.uniform(-0.5, 0.5)
    total_err = 0

    # Training loop
    for iteration in range(ni):
        total_err = 0
        for i in range(len(X_train)):
            net = np.dot(weights, X_train[i]) + bias
            output = soft_act_test(net)
            err = Y_train[i] - output
            total_err += err**2
            learn = alpha * err

            # Update weights with derivative of sigmoid
            for k in range(n):
                weights[k] = weights[k] + learn * X_train[i][k] * output * (1 - output)
            bias += learn * output * (1 - output)

        # Check stopping criterion
        if total_err < epsilon:
            print(f"Converged at iteration {iteration + 1}")
            break

    print(f"Training Total Error (TE): {total_err:.6f}")
    print(f"Iterations completed: {iteration + 1}")

    # Testing phase
    out = []
    for i in range(len(X_test)):
        net = np.dot(weights, X_test[i]) + bias
        output = soft_act_test(net)
        out.append(output)

    return X_train, Y_train, X_test, Y_test, weights, bias, out, Y_test


def hard_activation(X, Y, test_percent, ni, terr):
    """Train perceptron with hard unipolar activation function"""
    alpha = 0.07
    epsilon = terr
    n = X.shape[1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_percent, random_state=42, stratify=Y
    )

    # Initialize the neuron with random weights between -0.5 and 0.5
    np.random.seed(42)
    weights = np.random.uniform(-0.5, 0.5, n)
    bias = np.random.uniform(-0.5, 0.5)
    total_err = 0

    # Training loop
    for iteration in range(ni):
        total_err = 0
        for i in range(len(X_train)):
            net = np.dot(weights, X_train[i]) + bias
            output = hard_act_test(net)
            err = Y_train[i] - output
            total_err += err**2
            learn = alpha * err

            # Update weights
            for k in range(n):
                weights[k] = weights[k] + learn * X_train[i][k]
            bias += learn

        # Check stopping criterion
        if total_err < epsilon:
            print(f"Converged at iteration {iteration + 1}")
            break

    print(f"Training Total Error (TE): {total_err:.6f}")
    print(f"Iterations completed: {iteration + 1}")

    # Testing phase
    out = []
    for i in range(len(X_test)):
        net = np.dot(weights, X_test[i]) + bias
        output = hard_act_test(net)
        out.append(output)

    return X_train, Y_train, X_test, Y_test, weights, bias, out, Y_test


def plotting_data(X_tr, Y_tr, X_te, Y_te, w, b, group, percent, activation):
    """Plot training and testing data with decision boundary"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot training data
    small_car_1 = X_tr[Y_tr == 0]
    big_car_1 = X_tr[Y_tr == 1]

    ax1.scatter(small_car_1[:, 0], small_car_1[:, 1], alpha=0.6, label="Small Cars", s=50)
    ax1.scatter(big_car_1[:, 0], big_car_1[:, 1], alpha=0.6, label="Big Cars", s=50)

    # Plot decision boundary for training data
    x1_min, x1_max = X_tr[:, 0].min() - 0.1, X_tr[:, 0].max() + 0.1
    x1_line = np.linspace(x1_min, x1_max, 100)

    if abs(w[1]) > 1e-10:
        x2_line = -(w[0] * x1_line + b) / w[1]
        ax1.plot(x1_line, x2_line, "k-", linewidth=2, label="Decision Boundary")

    ax1.set_xlim(x1_min, x1_max)
    ax1.set_ylim(X_tr[:, 1].min() - 0.1, X_tr[:, 1].max() + 0.1)
    ax1.set_xlabel("Normalized Price", fontsize=11)
    ax1.set_ylabel("Normalized Weight", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Training Data of {group}: {activation}\n{percent}", fontsize=12)

    # Plot testing data
    small_car_2 = X_te[Y_te == 0]
    big_car_2 = X_te[Y_te == 1]

    ax2.scatter(small_car_2[:, 0], small_car_2[:, 1], alpha=0.6, label="Small Cars", s=50)
    ax2.scatter(big_car_2[:, 0], big_car_2[:, 1], alpha=0.6, label="Big Cars", s=50)

    # Plot decision boundary for testing data
    x3_min, x3_max = X_te[:, 0].min() - 0.1, X_te[:, 0].max() + 0.1
    x3_line = np.linspace(x3_min, x3_max, 100)

    if abs(w[1]) > 1e-10:
        x4_line = -(w[0] * x3_line + b) / w[1]
        ax2.plot(x3_line, x4_line, "k-", linewidth=2, label="Decision Boundary")

    ax2.set_xlim(x3_min, x3_max)
    ax2.set_ylim(X_te[:, 1].min() - 0.1, X_te[:, 1].max() + 0.1)
    ax2.set_xlabel("Normalized Price", fontsize=11)
    ax2.set_ylabel("Normalized Weight", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Testing Data of {group}: {activation}\n{percent}", fontsize=12)
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate filename based on parameters
    # Clean up group name and activation for filename
    group_clean = group.replace(" ", "")
    activation_short = "Hard" if "Hard" in activation else "Soft"
    percent_clean = percent.replace("%", "").replace(" ", "").replace(",", "_")
    
    filename = f"plots/{group_clean}_{activation_short}_{percent_clean}.png"
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    
    plt.show()


def confusion_mat(w, b, o, y_t, g, act, per):
    """Calculate and display confusion matrix with all rates"""
    # Convert outputs to binary (threshold at 0.5 for soft activation)
    o_con = np.where(np.array(o) >= 0.5, 1, 0)

    # Calculate confusion matrix elements
    TP = np.sum((y_t == 1) & (o_con == 1))
    TN = np.sum((y_t == 0) & (o_con == 0))
    FP = np.sum((y_t == 0) & (o_con == 1))
    FN = np.sum((y_t == 1) & (o_con == 0))

    # Calculate metrics
    total = len(y_t)
    accuracy = (TP + TN) / total if total > 0 else 0
    error_rate = 1 - accuracy
    
    # Calculate rates
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Sensitivity/Recall)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Confusion Matrix for {g} - {act}")
    print(f"{per}")
    print(f"{'='*70}")
    
    print(f"\n{'':20} Predicted Big (1)    Predicted Small (0)")
    print(f"Actual Big (1)       {TP:8d}               {FN:8d}")
    print(f"Actual Small (0)     {FP:8d}               {TN:8d}")
    
    print(f"\n{'Metrics:':20}")
    print(f"{'Accuracy:':25} {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'Error Rate:':25} {error_rate:.4f} ({error_rate*100:.2f}%)")
    print(f"{'True Positive Rate:':25} {TPR:.4f} ({TPR*100:.2f}%)")
    print(f"{'True Negative Rate:':25} {TNR:.4f} ({TNR*100:.2f}%)")
    print(f"{'False Positive Rate:':25} {FPR:.4f} ({FPR*100:.2f}%)")
    print(f"{'False Negative Rate:':25} {FNR:.4f} ({FNR*100:.2f}%)")
    print(f"{'Precision:':25} {precision:.4f} ({precision*100:.2f}%)")
    print(f"\nWeights: {w}")
    print(f"Bias: {b:.6f}")
    print(f"{'='*70}\n")


def sua_print(X, Y, per, n_ite, ep, grp, txt, test):
    """Soft Unipolar Activation - Train, Test, and Display Results"""
    print(f"\n{'#'*70}")
    print(f"# {grp} - {txt}")
    print(f"# {test}")
    print(f"{'#'*70}")
    
    X_tr, Y_tr, X_te, Y_te, weight, bias, o, y_t = soft_activation(X, Y, per, n_ite, ep)
    
    print(f"\nTraining samples: {len(Y_tr)} | Testing samples: {len(Y_te)}")
    
    confusion_mat(weight, bias, o, y_t, grp, txt, test)
    plotting_data(X_tr, Y_tr, X_te, Y_te, weight, bias, grp, test, txt)


def hua_print(X, Y, per, n_ite, ep, grp, txt, test):
    """Hard Unipolar Activation - Train, Test, and Display Results"""
    print(f"\n{'#'*70}")
    print(f"# {grp} - {txt}")
    print(f"# {test}")
    print(f"{'#'*70}")
    
    X_tr, Y_tr, X_te, Y_te, weight, bias, o, y_t = hard_activation(X, Y, per, n_ite, ep)
    
    print(f"\nTraining samples: {len(Y_tr)} | Testing samples: {len(Y_te)}")
    
    confusion_mat(weight, bias, o, y_t, grp, txt, test)
    plotting_data(X_tr, Y_tr, X_te, Y_te, weight, bias, grp, test, txt)


def main():
    """Main function to run all experiments"""
    # Dataset file paths
    d1 = "groupA.txt"
    d2 = "groupB.txt"
    d3 = "groupC.txt"

    # Create plots directory at the start
    os.makedirs('plots', exist_ok=True)
    print("Created 'plots' directory for saving figures...")

    # Load and normalize datasets
    print("Loading and normalizing datasets...")
    X1, Y1 = normalize(d1, "A")
    X2, Y2 = normalize(d2, "B")
    X3, Y3 = normalize(d3, "C")
    print("Datasets loaded successfully!\n")

    # Parameters
    ni = 5000  # Maximum number of iterations

    # Labels
    grp1, grp2, grp3 = "Group A", "Group B", "Group C"
    hd = "Hard Unipolar Activation Function"
    s = "Soft Unipolar Activation Function"
    sf = "75% Training, 25% Testing"
    tf = "25% Training, 75% Testing"

    print("\n" + "="*70)
    print("STARTING PERCEPTRON CLASSIFICATION EXPERIMENTS")
    print("="*70)

    # ========================================================================
    # HARD ACTIVATION FUNCTION
    # ========================================================================
    print("\n\n" + "*"*70)
    print("* PART 1: HARD UNIPOLAR ACTIVATION FUNCTION")
    print("*"*70)

    # Dataset A - 75% training, 25% testing
    hua_print(X1, Y1, 0.25, ni, 1e-5, grp1, hd, sf)

    # Dataset B - 75% training, 25% testing
    hua_print(X2, Y2, 0.25, ni, 40, grp2, hd, sf)

    # Dataset C - 75% training, 25% testing
    hua_print(X3, Y3, 0.25, ni, 700, grp3, hd, sf)

    # Dataset A - 25% training, 75% testing
    hua_print(X1, Y1, 0.75, ni, 1e-5, grp1, hd, tf)

    # Dataset B - 25% training, 75% testing
    hua_print(X2, Y2, 0.75, ni, 40, grp2, hd, tf)

    # Dataset C - 25% training, 75% testing
    hua_print(X3, Y3, 0.75, ni, 700, grp3, hd, tf)

    # ========================================================================
    # SOFT ACTIVATION FUNCTION
    # ========================================================================
    print("\n\n" + "*"*70)
    print("* PART 2: SOFT UNIPOLAR ACTIVATION FUNCTION")
    print("*"*70)

    # Dataset A - 75% training, 25% testing
    sua_print(X1, Y1, 0.25, ni, 1e-5, grp1, s, sf)

    # Dataset B - 75% training, 25% testing
    sua_print(X2, Y2, 0.25, ni, 40, grp2, s, sf)

    # Dataset C - 75% training, 25% testing
    sua_print(X3, Y3, 0.25, ni, 700, grp3, s, sf)

    # Dataset A - 25% training, 75% testing
    sua_print(X1, Y1, 0.75, ni, 1e-5, grp1, s, tf)

    # Dataset B - 25% training, 75% testing
    sua_print(X2, Y2, 0.75, ni, 40, grp2, s, tf)

    # Dataset C - 25% training, 75% testing
    sua_print(X3, Y3, 0.75, ni, 700, grp3, s, tf)

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("\nSummary:")
    print("- Total experiments run: 12 (6 hard + 6 soft)")
    print("- Datasets processed: A, B, C")
    print("- Train/test splits: 75-25 and 25-75")
    print("- Activation functions: Hard and Soft Unipolar")
    print("="*70)


if __name__ == "__main__":
    main()