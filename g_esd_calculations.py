# -*- coding: utf-8 -*-
"""
@author prajaktab2

"""
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.seasonal import STL

def check_G_values(Gs,Gc,inp,max_index):
    if Gs > Gc:
        print("{} is an outlier.G > Gcritical: {:.4f} > {:.4f} \n".format(inp[max_index],Gs,Gc))
        return True,max_index,inp[max_index]
    else:
        print("{} is not an outlier.G < Gcritical: {:.4f} < {:.4f} \n".format(inp[max_index],Gs,Gc))
        return False,0,0
    
def grubbs_stat(ts,algo):
    if algo=="G-ESD":
        return grubbs_stat_G_ESD(ts)
    elif algo=="S-ESD":
        return grubbs_stat_S_ESD(ts)
    elif algo=="S-H-ESD":
        return grubbs_stat_S_H_ESD(ts)
               
    
    
def grubbs_stat_S_ESD(ts):  
    stl = STL(ts, seasonal=13)
    res = stl.fit()
    residual = ts - res.seasonal - np.median(ts)
    return grubbs_stat_G_ESD(residual)
    
def grubbs_stat_S_H_ESD(ts):  
    stl = STL(ts, seasonal=13)
    res = stl.fit()
    residual = ts - res.seasonal - np.median(ts)
    return grubbs_stat_H_ESD(residual)    
            
        
def grubbs_stat_G_ESD(y):
    std_dev=np.std(y)
    avg_y=np.mean(y)
    abs_val_minus_avg = abs(y-avg_y)
    max_of_deviations=max(abs_val_minus_avg)
    max_ind=np.argmax(abs_val_minus_avg)
    Gcal=max_of_deviations/std_dev
    return Gcal, max_ind

def grubbs_stat_H_ESD(ts):
    median = np.ma.median(ts)
    mad = np.ma.median(np.abs(ts - median))
    scores = np.abs((ts - median) / mad)
    max_idx = np.argmax(scores)        
    return scores[max_idx],max_idx

'''he critical value can be calculated using the percent point function (PPF) for 
a given significance level, such as 0.05 (95% confidence).
This function is available for the t distribution in SciPy'''

def calculate_critical_value(size,alpha):
    t_dist=stats.t.ppf(1-alpha/(2*size),size-2)
    numerator=(size-1)*np.sqrt(np.square(t_dist))
    denominator= np.sqrt(size) * np.sqrt(size -2 +np.square(t_dist))
    critical_value=numerator/denominator
    return critical_value