# Hyundai Predictive Maintenance Project

This project attempts to build a predictive maintenance model for Hyundai vehicles using a provided dataset. The goal was to predict whether a vehicle would have an "Anomaly Indication" based on sensor readings and its maintenance history.

## Project Structure

```
.
├── data/
│   └── cars_hyundai.csv
├── images/
│   ├── anomaly_distribution.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
│   └── ... (other plots)
├── models/
│   ├── best_random_forest_model.pkl
│   └── ... (other models)
├── src/
│   ├── 01_EDA_and_Preprocessing.py
│   ├── 02_model_training.py
│   ├── 03_hyperparameter_tuning.py
│   ├── 04_xgboost_modeling.py
│   └── 05_feature_importance.py
└── README.md
```

## Methodology

Our approach followed a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):** We first analyzed the dataset to understand its structure, check for missing values, and visualize the relationships between features. The target variable, `Anomaly Indication`, was found to be well-balanced.

2.  **Modeling:** We treated the problem as a supervised binary classification task. We experimented with several models:
    *   A baseline `RandomForestClassifier`.
    *   A hyperparameter-tuned `RandomForestClassifier` using `GridSearchCV`.
    *   An `XGBoost` classifier.

3.  **Evaluation:** Models were evaluated on a dedicated test set using standard classification metrics, including accuracy, precision, recall, and F1-score.

## Results and Conclusion

Despite a rigorous process of modeling and tuning, no model was able to achieve a performance significantly better than a random guess (50-57% accuracy).

The key insight came from a **feature importance analysis** performed on the best model. The analysis revealed that none of the provided features (`Engine Temperature`, `Brake Pad Thickness`, `Tire Pressure`, `Maintenance Type`) had strong predictive power. The feature importance scores were low and nearly evenly distributed across all features.

### Final Verdict

**The dataset in its current form is insufficient to build a reliable predictive maintenance model.**

The low model performance is not a failure of the modeling process, but rather a direct result of the features lacking a clear signal to distinguish between anomalous and non-anomalous events.

## Recommendations for Future Work

To successfully build a predictive model, a more comprehensive dataset would be required. We recommend collecting data with more predictive features, such as:

*   **Time-Series Data:** Sensor readings over a period of time leading up to a maintenance event.
*   **Additional Sensors:** Data from more components like oil viscosity, vibration sensors, or exhaust readings.
*   **Operational History:** Vehicle age, mileage, and driving conditions (e.g., city vs. highway).
*   **More Granular Labels:** A more detailed definition of what constitutes an "anomaly" or failure. 