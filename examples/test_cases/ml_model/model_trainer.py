import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
    
    def preprocess(self):
        # Example preprocessing: handling missing values and encoding categorical variables
        self.data.fillna(method='ffill', inplace=True)
        self.data['category'] = self.data['category'].astype('category').cat.codes
        return self.data

class ModelTrainer:
    def __init__(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor
    
    def train_model(self):
        # Preprocess the data
        preprocessed_data = self.data_preprocessor.preprocess()
        
        # Split the data into features and target variable
        X = preprocessed_data.drop('target', axis=1)
        y = preprocessed_data['target']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train a logistic regression model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
        
        # Visualize the results
        plt.figure(figsize=(10, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    # Example data
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    data_preprocessor = DataPreprocessor(df)
    model_trainer = ModelTrainer(data_preprocessor)
    model_trainer.train_model()