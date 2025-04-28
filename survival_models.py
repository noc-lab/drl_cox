import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored,cumulative_dynamic_auc
from sksurv.util import Surv

def aft_survival_analysis(train_data, X_test, y_test, duration_col='time', event_col='event'):
    """
    Perform survival analysis using the Accelerated Failure Time (AFT) model.

    Parameters:
        train_data (pd.DataFrame): Training dataset.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test target, with columns for duration and event indicator.
        duration_col (str): Column name for survival times in `train_data` and `y_test`.
        event_col (str): Column name for event indicators (1 for event, 0 for censoring).

    Returns:
        dict: A dictionary containing the fitted model, concordance index, predictions, and mean AUC.
    """

    # Ensure event column is boolean
    y_test[event_col] = y_test[event_col].astype(bool)

    # Fit the AFT model using Weibull distribution
    aft_model = WeibullAFTFitter()
    aft_model.fit(train_data, duration_col=duration_col, event_col=event_col)

    # Predict survival times for test set
    survival_times = aft_model.predict_median(X_test)

    # Calculate concordance index for performance evaluation
    c_index = concordance_index(y_test[duration_col], survival_times, y_test[event_col])

    # Compute Kaplan-Meier censoring survival function
    kmf = KaplanMeierFitter()
    kmf.fit(y_test[duration_col], event_observed=1 - y_test[event_col])  # Estimate censoring function

    # Get valid time points where censoring probability > 0
    censoring_survival = kmf.survival_function_
    valid_times = censoring_survival.index[censoring_survival["KM_estimate"] > 0].to_numpy()

    # Restrict time range within observed event times
    min_time = max(valid_times.min(), y_test[duration_col].min()) + 0.1
    max_time = min(valid_times.max(), y_test[duration_col].max()) - 0.1
    time_points = np.linspace(min_time, max_time, 100)

    # Filter out time points where censoring probability is zero
    time_points = np.array([t for t in time_points if kmf.predict(t) > 0])

    # Predict survival probabilities at valid time points
    survival_probabilities = pd.DataFrame(
        {t: aft_model.predict_survival_function(X_test, times=[t]).values.flatten() for t in time_points},
        index=X_test.index
    )

    # Convert y_test into sksurv format
    survival_data = Surv.from_dataframe(event_col, duration_col, y_test)

    # Ensure risk_scores matches y_test length
    risk_scores = -survival_times.values  # Convert Series to NumPy array

    # Compute time-dependent AUC only for valid time points
    if len(time_points) > 0:
        auc_scores = cumulative_dynamic_auc(survival_data, survival_data, risk_scores, time_points)[1]
        mean_auc = np.mean(auc_scores)
    else:
        mean_auc = np.nan  # If no valid time points, set AUC to NaN

    return {
        "model": aft_model,
        "concordance_index": c_index,
        "predicted_survival_times": survival_times,
        "predicted_survival_probabilities": survival_probabilities,
        "auc": mean_auc
    }



def random_survival_forest_analysis(X_train, y_trains, X_test, y_tests, n_estimators=100, min_samples_split=10, min_samples_leaf=4, max_features="sqrt", random_state=42):
    """
    Perform survival analysis using Random Survival Forest (RSF).

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test target.
        n_estimators (int): Number of trees in the forest.
        min_samples_split (int): Minimum samples required to split an internal node.
        min_samples_leaf (int): Minimum samples required to be at a leaf node.
        max_features (str or int): Number of features to consider when looking for the best split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the fitted model, concordance index, and predictions.
    """
    # Reformat y.
    y_train = Surv.from_arrays(event=y_trains.values[:,1], time=y_trains.values[:,0])
    y_test = Surv.from_arrays(event=y_tests.values[:,1], time=y_tests.values[:,0])
    #print(y_test[:10])

    # Initialize the Random Survival Forest model
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )

    # Fit the model to the training data
    rsf.fit(X_train, y_train)

    # Predict risk scores for the test set
    risk_scores = rsf.predict(X_test)

    # Calculate concordance index for performance evaluation
    # c_index = concordance_index_censored(
    #     event_indicator=y_test["event"],  # Censoring indicator (1 = event occurred, 0 = censored)
    #     event_time=y_test["time"],       # Survival or censoring time
    #     estimate=risk_scores             # Predicted risk scores
    # )
    c_index = concordance_index(y_test["time"],-risk_scores,y_test["event"])

    # Compute time-dependent AUC
    time_points = np.linspace(min(y_test["time"])+0.1, max(y_test["time"])-0.1, 100)
    auc_scores = cumulative_dynamic_auc(y_test, y_test, risk_scores, time_points)[1]

    # Compute mean AUC
    mean_auc = np.mean(auc_scores)

    return {
        "model": rsf,
        "concordance_index": c_index,
        "predicted_risk_scores": risk_scores,
        "auc": mean_auc
    }
