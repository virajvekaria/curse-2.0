import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Example preprocessing steps
        self.data = self.data.dropna()
        X = self.data.drop('target', axis=1)
        y = self.data['target']

        # Splitting the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        print("Data preprocessing completed.")

def main():
    data_path = 'path_to_your_data.csv'
    dp = DataPreprocessor(data_path)
    dp.load_data()
    dp.preprocess_data()

    # Example model training
    model = LogisticRegression(max_iter=200)
    model.fit(dp.X_train, dp.y_train)

    # Model evaluation
    y_pred = model.predict(dp.X_test)
    accuracy = accuracy_score(dp.y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(dp.y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(dp.y_test, y_pred))

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(dp.y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()