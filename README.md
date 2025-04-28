# drl_cox

**drl_cox** is a **Wasserstein-based Distributionally Robust Cox Model**. This model uses distributionally robust optimization (DRO) techniques to improve the robustness of survival analysis, particularly in the presence of distributional shifts.

### Features
- Implements a **distributionally robust version** of the **Cox Proportional Hazards model** using Wasserstein distance.
- Utilizes **Random Survival Forests** (RSF)  **Accelerated Failure Time Model** (AFT) and **Penalized Cox Models** as baseline models.
- Inject contamination to the test datas to simulate distributional shift and compare the performance.

### Requirements
- numpy==1.24.0
- pandas
- lifelines
- sksurv
- torch
- cvxpy
- clarabel/mosek
- scipy
- sklearn

### Usage
To run the model with WHAS500, use the following command:

`python eval.py`

To run the model with other dataset, use the command line with the following structure:

`python eval.py --data_path /path/to/data.csv --output_path /path/to/results/`
