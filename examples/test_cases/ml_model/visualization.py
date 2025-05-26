import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(y_true, y_pred):
    """
    Plots the true vs predicted values for a classification model.
    
    Parameters:
    - y_true: True labels (numpy array or list)
    - y_pred: Predicted labels (numpy array or list)
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1]) > 1.5
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Target'] = y
    
    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df[['Feature1', 'Feature2']], df['Target'], test_size=0.2, random_state=42)
    
    # Train a simple logistic regression model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Plot results
    plot_results(y_test, y_pred)