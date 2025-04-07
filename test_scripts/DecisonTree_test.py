import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.DecisionTree import DecisionTree
import numpy as np
from data import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprcess_data(train_features, train_labels, test_features, test_labels):
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )
    return train_features, val_features, train_labels, val_labels, test_features, test_labels

def train_model(train_features, train_labels, val_features, val_labels):
    model = DecisionTree(max_depth=10)
    model.fit(train_features, train_labels)

    # Evaluate on validation set
    val_predictions = model.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # save the model
    # model.save("decision_tree_model.pkl")

# def test_model(model, test_features, test_labels):
    # Evaluate on test set
    test_predictions = model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels = load_data.fetch_data()
    train_features, val_features, train_labels, val_labels, test_features, test_labels = preprcess_data(train_features, train_labels, test_features, test_labels)
    train_model(train_features, train_labels, val_features, val_labels)






