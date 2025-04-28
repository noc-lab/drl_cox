import cvxpy as cp
import numpy as np
import time
import lifelines
import pandas as pd
from scipy.special import logsumexp
import clarabel
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from lifelines.calibration import survival_probability_calibration
from metrics import *
from Hu_Chen_model import *
from load_data import *
from drl_cox import *
from contaminate import *
import copy
import sksurv
from sksurv import datasets
import argparse
import warnings
from survival_models import aft_survival_analysis, random_survival_forest_analysis
from sklearn import preprocessing

def compare(df,ver=False,epsilons=[0.5],norm=[2],simple=3,outliers_ratio=[0.1],
                   outliers_intensity=[0.1],solver='MOSEK',random=False,num=1,gpu_devices=[0,1,2],
            train_size=400,val_size=500,test_size=500,rounds=5,noise_type="perturb"):
    # The data's last column should be event indicator, and the last 2nd column should be duration.
    # Return a 5D list cindex, where cindex[i][j][k][l]= cindex of regular cox, 
    # penalized cox and dro cox, with the parameter norm[i], outliers_ratio[j] and outliers_intensity[k].
    # Be sure to input num, which is the number of numerical features.
    # Input train, validation and test datasets, output performance of all models.
    datas=df.values
    datas=datas.astype('float')
    ans1,ans2=[],[]
    cols= list(df.columns)#list(range(1,1000))
    n,m=train_size,len(datas[0])-2
    
    y_max=datas.max(axis=0)[m]
    y_min=datas.min(axis=0)[m]

    trains,vals,tests=[],[],[]
    coxbs,pcoxbs1,pcoxbs2,pcoxbs3,drobs=[],[],[],[],[]
    i=0
    while i<rounds:
        # Shuffule the datas and split it into train, valid and test sets.
        np.random.shuffle(datas)
        train,val,test=copy.deepcopy(datas[:n]),copy.deepcopy(datas[n:n+val_size]),copy.deepcopy(datas[n+val_size:n+val_size+test_size])
        
        # Normalize numerical features
        scaler = preprocessing.MinMaxScaler()
        X = copy.deepcopy(train[:,:num])
        X = scaler.fit_transform(X)
        train[:,:num] = copy.deepcopy(X)
        val[:,:num] = scaler.transform(val[:,:num])
        test[:,:num] = scaler.transform(test[:,:num])

        # Normalize target variable (y_col)
        y_mean,y_std = train[:,-2].mean(),train[:,-2].std()
        train[:,-2] = copy.deepcopy((train[:,-2] - y_min+1) * 2 / y_std)
        val[:,-2] = copy.deepcopy((val[:,-2] - y_min+1) * 2 / y_std)
        test[:,-2] = copy.deepcopy((test[:,-2] - y_min+1) * 2 / y_std)
        temp=copy.deepcopy(train)

        trains.append(train)
        vals.append(val)
        tests.append(test)

        try:
            # Create regular Cox model.
            df=pd.DataFrame(temp,columns=cols)
            cph = CoxPHFitter()
            cph.fit(df, duration_col=cols[m], event_col=cols[m+1])
            cox_b=list(cph.params_)

            s=[]
            x,y,z=temp[:,:m],temp[:,m],temp[:,-1]
            for j in range(len(temp)):
                if z[j]==0:
                    s.append(0)
                else:
                    s.append(np.log(np.exp(np.dot(cox_b,x[j]))+sum([np.exp(np.dot(cox_b,x[j])) for j in range(j,len(temp))])) - np.dot(x[j],cox_b))
            s=np.array(s)
            # Use cvxpy
            bs=drl_cox_cvxpy(temp,simple=simple,norm=2,epsilons=epsilons,solver="MOSEK",b_0=np.append(cox_b,0),s_0=s,random=random)

            # Compute penalized Cox models.
            pcph1 = CoxPHFitter(penalizer=0.5,l1_ratio=0)
            pcph1.fit(df, duration_col=cols[m], event_col=cols[m+1])
            pcox_b1=list(pcph1.params_)
            
            pcph2 = CoxPHFitter(penalizer=0.5,l1_ratio=1)
            pcph2.fit(df, duration_col=cols[m], event_col=cols[m+1])
            pcox_b2=list(pcph2.params_)

            pcph3 = CoxPHFitter(penalizer=0.5,l1_ratio=0.5)
            pcph3.fit(df, duration_col=cols[m], event_col=cols[m+1])
            pcox_b3=list(pcph3.params_)
            print(f"Round {i+1} training complete.")
        except:
            print("Solver failed for once.")
            continue
        coxbs.append(cox_b)
        pcoxbs1.append(pcox_b1)
        pcoxbs2.append(pcox_b2)
        pcoxbs3.append(pcox_b3)
        drobs.append(bs)
        i+=1

    for lp in norm:
        for r in outliers_ratio:
            for p_var in outliers_intensity:
                print(f"Current norm = {lp}, outlier ratio = {r}, perturbation variance = {p_var}.")
                # pa1 = Ridge Cox, pa2 = Lasso Cox, pa3 = Elastic Net Cox

                i=1
                ca,da,pa1,pa2,pa3,ha,aa,ra=0,0,0,0,0,0,0,0
                cauc,dauc,pauc1,pauc2,pauc3,hauc,aauc,rauc=0,0,0,0,0,0,0,0
                t=time.time()

                while i<=rounds:
                    cox_b,pcox_b1,pcox_b2,pcox_b3,bs=coxbs[i-1],pcoxbs1[i-1],pcoxbs2[i-1],pcoxbs3[i-1],drobs[i-1]
                    train,val,test=copy.deepcopy(trains[i-1]),copy.deepcopy(vals[i-1]),copy.deepcopy(tests[i-1])
                    temp=copy.deepcopy(train)

                    # Now, apply contamination (noise) to the validation and test data
                    n_val = len(val)
                    n_test = len(test)

                    # Merge into one NumPy array
                    combined = np.vstack([val, test])

                    # Apply noise once on combined
                    if noise_type == "perturb":
                        combined_noisy = perturb(combined, var=p_var, num_col=num, ratio=r)
                    elif noise_type == "shift":
                        combined_noisy = shift(combined, var=p_var, num_col=num, ratio=r)
                    else:
                        combined_noisy = combined
                    np.random.shuffle(combined_noisy)

                    # Split back into val and test
                    val  = combined_noisy[:n_val]
                    test = combined_noisy[n_val: n_val + n_test]
                    
                    df_val = pd.DataFrame(test, columns=cols)
                    df=pd.DataFrame(temp,columns=cols)

                    try:
                        # Compute aft model.
                        aft = aft_survival_analysis(
                            df,df_val[cols[:-2]],df_val[cols[-2:]],duration_col=cols[-2], event_col=cols[-1]
                            )
                        temp_aa=aft['concordance_index']
                        temp_aauc=aft['auc']
                        # print(f"The AFT Cindex here is {temp_aa}.")

                        # Compute RSF model.
                        rsf = random_survival_forest_analysis(df[cols[:-2]],df[cols[-2:]],df_val[cols[:-2]],df_val[cols[-2:]])
                        temp_ra=rsf['concordance_index']
                        temp_rauc=rsf['auc']
                        # print(f"The RSF Cindex here is {temp_ra}, and RSF auc here is {temp_rauc}.")

                        # Compute Hu and Chen's sample splitting Cox model.
                        best_hauc=0
                        for eps in epsilons[1:]:
                            hu_b=Hu_solve(temp,epsilon=eps)
                            # if temp_ha<=cindex(hu_b,val):
                            if best_hauc<=auc(hu_b,val):
                                best_hauc=auc(hu_b,val)
                                temp_hauc=auc(hu_b,test)
                                temp_ha=cindex(hu_b,test)
                        
                    except Exception as e:
                        # prints the exception’s message
                        print("Error occurred:", e)
                        # if you want the full traceback:
                        import traceback
                        traceback.print_exc()
                        # raise
                        continue

                    best_dauc=0
                    for b_ind in range(len(bs)):
                        temp_dauc=auc(bs[b_ind],val)
                        if temp_dauc>best_dauc:
                            best_dauc = temp_dauc
                            dro_b=bs[b_ind]
                            epsilon=epsilons[b_ind]
                    temp_ca=cindex(cox_b,test)
                    temp_cauc=auc(cox_b,test)
                    temp_da = cindex(dro_b,test)
                    temp_dauc = auc(dro_b,test)

                    print(f"Best epsilon={epsilon}.")
                    
                    temp_pa1=cindex(pcox_b1,test)
                    temp_pauc1=auc(pcox_b1,test)
                    temp_pa2=cindex(pcox_b2,test)
                    temp_pauc2=auc(pcox_b2,test)
                    temp_pa3=cindex(pcox_b3,test)
                    temp_pauc3=auc(pcox_b3,test)

                    ca+=temp_ca/rounds
                    da+=temp_da/rounds
                    pa1+=temp_pa1/rounds
                    pa2+=temp_pa2/rounds
                    pa3+=temp_pa3/rounds
                    ha+=temp_ha/rounds
                    aa+=temp_aa/rounds
                    ra+=temp_ra/rounds

                    cauc+=temp_cauc/rounds
                    dauc+=temp_dauc/rounds
                    pauc1+=temp_pauc1/rounds
                    pauc2+=temp_pauc2/rounds
                    pauc3+=temp_pauc3/rounds
                    hauc+=temp_hauc/rounds
                    aauc+=temp_aauc/rounds
                    rauc+=temp_rauc/rounds

                    # Compute time-dependent auc.
                    i+=1
        
                print(f"Regular Cox: C-index={ca}, auc={cauc}.")
                print(f"Ridge Cox: C-index={pa1}, auc={pauc1}.")
                print(f"Lasso Cox: C-index={pa2}, auc={pauc2}.")
                print(f"Elastic Net Cox: C-index={pa3}, auc={pauc3}.")
                print(f"DRO Cox: C-index={da}, auc={dauc}.")
                print(f"Hu and Chen's model: C-index={ha}, auc={hauc}.")
                print(f"AFT's model: C-index={aa}, auc={aauc}.")
                print(f"RSF model: C-index={ra}, auc={rauc}.")
                print(f"This round took {time.time()-t} seconds.")
                ans1.append([ca,pa1,pa2,pa3,ha,aa,ra,da])
                ans2.append([cauc,pauc1,pauc2,pauc3,hauc,aauc,rauc,dauc])
    cind=np.array(ans1)
    aucs=np.array(ans2)
    
    return cind,aucs


if __name__ == "__main__":
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='DRL for Survival Analysis')
    parser.add_argument('--data', type=str, default="Whas500")
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--train_size',type=int,default=300)
    parser.add_argument('--val_size',type=int,default=100)
    parser.add_argument('--test_size',type=int,default=100)
    parser.add_argument('--y',type=str,default='col_time')
    parser.add_argument('--d',type=str,default='sab')
    parser.add_argument('--noise',type=str,default='shift')
    args = parser.parse_args()
    out = str(args)
    print(out)
    rounds = 10

    if args.data == 'Whas500':   
        df,num=process_whas500()
        outr =[1]
        outint = [1,2,3,4,5,6,7,8]
        rounds = 50
    else:
        df=pd.read_csv(args.data_path)
        df,num = load_dataset(df,y_col=args.y,d_col=args.d)
    data=df.values


    print(f"The censor rate is {1-np.mean(data[:,-1])}.")
    cind,aucs = compare(df,epsilons=[0,1e-3,5e-3,0.01,0.05,0.09,0.1,0.11,0.2,0.3,0.4,0.5,1],norm=[2],simple=3,
        outliers_ratio=outr,
        outliers_intensity=outint,
        random=False,num=num,
    train_size=args.train_size,val_size=args.val_size,test_size=args.test_size,rounds=rounds,noise_type=args.noise)
    
    # Printing the lists with commas to easily copy and paste them as new arrays
    print(f"cind = {cind.tolist()}")
    print(f"auc = {aucs.tolist()}")