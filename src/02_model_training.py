import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocesses the data for modeling."""
    # Separate target variable
    X = df.drop('Anomaly Indication', axis=1)
    y = df['Anomaly Indication']

    # One-hot encode the 'Maintenance Type' feature
    categorical_features = ['Maintenance Type']
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Apply encoder
    encoded_features = pd.DataFrame(one_hot_encoder.fit_transform(X[categorical_features]))
    encoded_features.columns = one_hot_encoder.get_feature_names_out(categorical_features)
    
    # Drop original categorical feature and concatenate encoded ones
    X = X.drop(categorical_features, axis=1)
    X = pd.concat([X, encoded_features], axis=1)
    
    return X, y

def train_and_evaluate_model(X, y):
    """Splits data into train, dev, and test sets, then trains and evaluates a model."""
    # First split: 70% train, 30% temporary set (for dev and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Second split: Split the temporary set into dev and test sets (50% each)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("Dataset split sizes:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Development set: {len(X_dev)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the development set
    print("\n--- Development Set Evaluation ---")
    y_dev_pred = model.predict(X_dev)
    print(f"Accuracy: {accuracy_score(y_dev, y_dev_pred):.4f}")
    print(classification_report(y_dev, y_dev_pred))

    # Final evaluation on the test set
    print("\n--- Test Set Evaluation ---")
    y_test_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred))
    
    # Save the model
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("\nModel saved to models/random_forest_model.pkl")

def main():
    """Main function to run the model training script."""
    print("Loading data...")
    df = load_data('data/cars_hyundai.csv')
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    print("Training model...")
    train_and_evaluate_model(X, y)
    
    print("\nModel training script finished.")

if __name__ == '__main__':
    main() 