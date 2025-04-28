import cvxpy as cp
import numpy as np
import time
import lifelines
import pandas as pd
import logging


def drl_cox_cvxpy(Tr,simple=5,norm=2,epsilons=[0.05],solver="CLARABEL",b_0=None,s_0=None,random=False):
    #T=temp[(-temp)[:, m].argsort()]
    bs=[]
    n,m=len(Tr),len(Tr[0])-2
    T=Tr[(-Tr)[:, m].argsort()]
    # Set variables and constraints. 
    x,y,d=T[:,:m],T[:,m],T[:,m+1]
    s=cp.Variable(n)
    bx=cp.Variable()
    b=cp.Variable(m+1)
    eps=cp.Parameter(nonneg=True)
    constraints=[]
    if b_0 is not None and s_0 is not None:
        b.value=b_0
        s.value=s_0
    for i in range(n):
        constraints+=[bx>=b[:m]@x[i].T]
    for i in range(n):
        if not d[i]:
            constraints+=[s[i] >= 0]
            continue
        k=0
        P=np.vstack([x[i],x])
        term=min(n,i+simple) if simple else n
        if random:
            constraints+=[s[i]>=(cp.log_sum_exp(b[:m]@P[:i+2].T-bx)+bx-b[:m]@x[i].T)]
            while True:
                r_sample = np.random.poisson(simple-1)
                if r_sample <= n-i:
                    break
            rds=np.random.choice(np.arange(i, n), size=r_sample, replace=False)

            for j in rds:
                constraints+=[s[i]>=(cp.log_sum_exp(b[:m]@P[:j+2].T-bx)+bx-b[:m]@x[i].T)-b[m]*(y[i]-y[j])]
        else:
            for j in range(i,term):
                constraints+=[s[i]>=(cp.log_sum_exp(b[:m]@P[:j+2].T-bx)+bx-b[:m]@x[i].T)-b[m]*(y[i]-y[j])]
                #constraints+=[s[i]>=(cp.log_sum_exp(b[:m]@x[:j+1].T-bx)+bx-b[:m]@x[i].T)-b[m]*(y[i]-y[j])]
    prob = cp.Problem(cp.Minimize(cp.norm(b,norm)*eps+cp.sum(s)/n),constraints)
    for ep in epsilons:
        eps.value=ep
        t1=time.time()
        prob.solve(solver=solver)
        bs.append(b.value[:m])
        t2=time.time()-t1
        #print(f"Time = {t2}s, norm={norm}, epsilon={ep}.")
    return np.array(bs)

