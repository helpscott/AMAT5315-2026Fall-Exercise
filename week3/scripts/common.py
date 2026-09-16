import json, pathlib, math
from collections import defaultdict
import numpy as np

def rows(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]
def by_temp(rs):
    d=defaultdict(list)
    for r in rs: d[float(r['T'])].append(r)
    return dict(sorted(d.items()))
def block_values(x, nblock):
    x=np.asarray(x,float); n=len(x)//nblock
    if n<1:return np.array([])
    return np.array([x[i*n:(i+1)*n].mean() for i in range(nblock)])
def tau_int(x):
    x=np.asarray(x,float); x=x-x.mean(); var=np.dot(x,x)/len(x)
    if var==0:return .5
    total=.5
    maxlag=min(len(x)-1, max(10, 6*int(math.ceil(total))))
    for k in range(1,len(x)):
        rho=np.dot(x[:-k],x[k:])/(len(x)-k)/var
        if rho<=0: break
        total += rho
        if k > 6*total: break
    return max(.5,total)
