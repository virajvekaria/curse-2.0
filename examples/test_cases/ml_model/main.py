import random

# Generate synthetic data for demonstration purposes
def generate_data(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
        if (x1 + x2) > 0:
            y.append(1)
        else:
            y.append(0)
        X.append([x1, x2])
    return X, y

# DataPreprocessor
def preprocess_data(X):
    return X

# ModelTrainer
def train_model(X_train, y_train):
    weights = [random.uniform(-1, 1) for _ in range(len(X_train[0]) + 1)]
    learning_rate = 0.1
    epochs = 1000
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train[i]
            z = sum(w * xi for w, xi in zip(weights, [1] + x))
            y_pred = 1 / (1 + exp(-z))
            
            error = y_true - y_pred
            weights[0] += learning_rate * error
            for j in range(len(x)):
                weights[j+1] += learning_rate * error * x[j]
    
    return weights

# EvaluationMetrics
def evaluate_model(model, X_test, y_test):
    predictions = []
    for x in X_test:
        z = sum(w * xi for w, xi in zip(model, [1] + x))
        y_pred = 1 / (1 + exp(-z))
        predictions.append(round(y_pred))
    
    accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
    return accuracy

# Visualization
def visualize_results(y_test, y_pred):
    pass  # No visualization in this simple example

def main():
    # Generate synthetic data
    X, y = generate_data(100)
    
    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()