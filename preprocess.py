import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col}")
    
    # Split into features (X) and target (y)
    # The target variable is 'y' (has the client subscribed a term deposit?)
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Split into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("bank_marketing.csv")
    # Store processed data for model training
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    print("Processed data saved.")
