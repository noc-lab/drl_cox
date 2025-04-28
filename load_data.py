import cvxpy as cp
import numpy as np
import time
import lifelines
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from sksurv import datasets
from pycox.datasets import support

def load_dataset(data,y_col='col_time',d_col='sab'):
    cols=list(data.columns)
    features=[col for col in cols if col not in [y_col,d_col]]
    df = data[features + [y_col, d_col]]
    
    categorical_columns = df.select_dtypes(include='float')

    # Find columns with exactly two unique float values
    binary_float_columns = [col for col in categorical_columns.columns if categorical_columns[col].nunique() == 2]

    # Output the result

    for col in binary_float_columns:
        unique_values = df[col].unique()
        # Sort the unique values so the smaller one becomes 0 and the larger one becomes 1
        sorted_values = sorted(unique_values)
        # Replace the values with 0 and 1
        df[col] = df[col].replace({sorted_values[0]: 0, sorted_values[1]: 1})
    cols=list(df.columns)
    cat_cols=binary_float_columns
    cat_cols.remove(d_col) #cat_cols.remove("pregnant")
    cat_cols.append(y_col)
    cat_cols.append(d_col) #cat_cols.append("pregnant")
    num_cols=[col for col in cols if col not in cat_cols]
    num=len(num_cols)
    print(f"There are {num} numerical features at front.")
    df=df[num_cols+cat_cols]

    return df,num

def process_whas500():
    x,y=datasets.load_whas500()
    ytime,death=[],[]
    for p in y:
        ytime.append(p[1])
        death.append(int(p[0]))
    x['ytime']=ytime
    x['death']=death
    y_col,d_col='ytime','death'
    data=x
    cols=list(data.columns)
    features=[col for col in cols if col not in [y_col,d_col]]
    df = data[features + [y_col, d_col]]

    cat_cols=[]
    # Find columns with exactly two unique float values
    for col in features:
        unique_values = df[col].unique()
        #print(col,len(unique_values))
        if len(unique_values)==2:
            cat_cols.append(col)
    cat_cols.append(y_col)
    cat_cols.append(d_col)
    num_cols=[col for col in features if col not in cat_cols]
    num=len(num_cols)
    print(f"There are {num} numerical features at front.")
    df=df[num_cols+cat_cols]
    
    return df,num

if __name__ == "__main__":
    df,num=process_whas500()
    df = df.astype(float)
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-detect width for better formatting
    pd.set_option('display.max_colwidth', None)  # No truncation of cell contents
    print(df.describe())