# Hyundai Predictive Maintenance Project: A Case Study

This project documents my process of attempting to build a predictive maintenance model for Hyundai vehicles. The initial goal was to predict whether a vehicle would have an "Anomaly Indication" based on a provided dataset of sensor readings and maintenance history.

Ultimately, this project serves as a practical example of a common and important scenario in data science: **sometimes, the most valuable insight is discovering that the available data is not sufficient to solve the problem.**

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

## My Approach

My approach followed a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):** I first analyzed the dataset to understand its structure, check for missing values, and visualize the relationships between features. I found that the target variable, `Anomaly Indication`, was well-balanced, which was a promising start.

2.  **Modeling:** I treated the problem as a supervised binary classification task and experimented with several models to find a predictive pattern:
    *   A baseline `RandomForestClassifier`.
    *   A hyperparameter-tuned `RandomForestClassifier` using `GridSearchCV`.
    *   An `XGBoost` classifier, a more powerful gradient boosting model.

3.  **Evaluation:** I evaluated each model on a dedicated test set using standard classification metrics, including accuracy, precision, recall, and F1-score, to ensure the performance measurement was robust.

## Results and Final Conclusion

My modeling process confirmed the initial hypothesis that the data lacks predictive power. Despite a rigorous process of modeling and tuning, no model was able to achieve a performance significantly better than a random guess.

Here is a summary of the test set performance for each model:

| Model | Accuracy | Weighted Avg F1-Score |
| :--- | :---: | :---: |
| Baseline Random Forest | 57.0% | 0.57 |
| Tuned Random Forest | 55.0% | 0.55 |
| XGBoost | 49.1% | 0.48 |

Notably, the performance slightly *decreased* as the model complexity increased. The most powerful model, XGBoost, produced the worst result. This is a strong indicator that the models were attempting to fit to noise rather than a true underlying signal.

The key insight came from a **feature importance analysis** I performed on the best model. The analysis clearly showed that none of the provided features (`Engine Temperature`, `Brake Pad Thickness`, `Tire Pressure`, `Maintenance Type`) had strong predictive power.

### The Verdict

My conclusion is that **the dataset, in its current form, is insufficient to build a reliable predictive maintenance model.**

This is not a failure of the modeling process, but a crucial finding about the data itself. The features available do not contain a strong enough signal to allow a machine learning model to distinguish between anomalous and non-anomalous events.

## Learning from this Project

This project highlights a fundamental truth in machine learning: the quality and predictive power of your data are more important than the complexity of your model. It demonstrates that a thorough and methodical process can lead to the important conclusion that a problem is not solvable without better data.

For this specific problem, I would recommend the following to move forward:

*   **Acquire Time-Series Data:** Sensor readings over a period of time leading up to a maintenance event would be far more predictive.
*   **Collect Data from Additional Sensors:** Information from components like oil viscosity sensors, vibration sensors, or exhaust readings could provide the necessary signal.
*   **Include Operational History:** Vehicle age, mileage, and driving conditions are critical context that is currently missing.
*   **Use More Granular Labels:** A more detailed definition of what constitutes an "anomaly" or failure would help clarify the target for the model. 