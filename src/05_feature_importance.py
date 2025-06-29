import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    """Preprocesses the data for modeling."""
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

def plot_feature_importance(model, feature_names):
    """Plots the feature importance of the trained model."""
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    print("Feature importance plot saved to images/feature_importance.png")
    plt.show()

def main():
    """Loads the model and plots feature importance."""
    print("Loading the best trained model...")
    # We need to preprocess data to get the feature names in the correct order
    df = pd.read_csv('data/cars_hyundai.csv')
    X, _ = preprocess_data(df)
    
    # Load the saved model
    model = joblib.load('models/best_random_forest_model.pkl')
    
    print("Plotting feature importances...")
    plot_feature_importance(model, X.columns)
    
    print("\nFeature importance analysis finished.")

if __name__ == '__main__':
    main() 