import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

def auc(b, test_data):
    """
    Compute time-dependent AUC for a Cox proportional hazards model.
    
    Parameters:
    - b: numpy array, estimated Cox model coefficients.
    - test_data: 2D list with columns [X1, X2, ..., Xp, "time", "event"].
      - "time": survival time
      - "event": event indicator (1 if event occurred, 0 for censored)
      - X1, X2, ..., Xp: covariates used in the Cox model
    
    Returns:
    - time_grid: Time points used for AUC computation.
    - auc_values: Corresponding AUC values at each time point.
    """
    # Extract survival times, event indicators, and covariates
    survival_times = np.array(test_data)[:,-2]
    event_indicators = np.array(test_data)[:,-1]
    X_test = np.array(test_data)[:,:-2]  # Covariates
    
    # Compute risk scores (linear predictor)
    risk_scores = np.dot(X_test, b)

    # Convert test survival data into structured array
    survival_test = Surv.from_arrays(event=event_indicators.astype(bool), time=survival_times)

    # Convert train survival data into structured array (used for cumulative_dynamic_auc)
    survival_train = survival_test

    times = np.sort(np.unique(np.array(test_data)[:, -2]))  # Extract unique times and sort them
    time_grid = np.linspace(times.min()+0.5, times.max()-0.5, min(10, len(times)))  # Generate grid with up to 10 points

    # Compute cumulative dynamic AUC
    times, auc_values = cumulative_dynamic_auc(survival_train, survival_test, risk_scores, time_grid)

    # Compute mean AUC
    mean_auc = np.mean(auc_values)

    return mean_auc

def cindex(b,datas):
    n,m = len(datas),len(datas[0])-2
    total,correct=0,0
    for i in range(n):
        if not datas[i][-1]:
            continue
        ri,ti=np.dot(b,datas[i][:m]),datas[i][m]
        for j in range(n):
            rj,tj=np.dot(b,datas[j][:m]),datas[j][m]
            if ti<tj:
                total+=1
                if ri>=rj:
                    correct+=1-0.5*int(ri==rj)
    return correct/total
    
