import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocesses the data for modeling."""
    X = df.drop('Anomaly Indication', axis=1)
    y = df['Anomaly Indication']
    
    categorical_features = ['Maintenance Type']
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    encoded_features = pd.DataFrame(one_hot_encoder.fit_transform(X[categorical_features]))
    encoded_features.columns = one_hot_encoder.get_feature_names_out(categorical_features)
    
    X = X.drop(categorical_features, axis=1)
    X = pd.concat([X, encoded_features], axis=1)
    
    return X, y

def tune_hyperparameters(X, y):
    """Performs hyperparameter tuning using GridSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize the model and GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    print("Starting hyperparameter tuning with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    # Evaluate the best model found by the grid search
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n--- Test Set Evaluation with Best Model ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    joblib.dump(best_model, 'models/best_random_forest_model.pkl')
    print("\nBest model saved to models/best_random_forest_model.pkl")

def main():
    """Main function to run the hyperparameter tuning script."""
    print("Loading data...")
    df = load_data('data/cars_hyundai.csv')
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    tune_hyperparameters(X, y)
    
    print("\nHyperparameter tuning script finished.")

if __name__ == '__main__':
    main() 