import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    print("Loading preprocessed data...")
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()
    
    # Initialize and train the Decision Tree Classifier
    print("Training Decision Tree Classifier...")
    # Using max_depth and min_samples_split to prevent overfitting
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = clf.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")
    
    # Decision Tree Visualization
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree.png")
    print("Decision tree visualization saved as decision_tree.png")

if __name__ == "__main__":
    train_and_evaluate()
