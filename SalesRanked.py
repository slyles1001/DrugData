# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:03:09 2018

@author: Jack
Data ordered by EU
"""

import scipy.io as spio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import time
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from numpy.linalg import LinAlgError
from datetime import datetime

def impmat(fname = 'M_processed.mat', writ = True):
    ''' import matlab crap, and turn it to pickles (or return panda df)'''
    mat = spio.loadmat(fname, squeeze_me=True)
    M = mat['M'] # data comes in as a bunch of TRASH, gotta make it PANDA
    #print(mat["__header__"])
    #print(mat["__version__"])
    #print(mat["__globals__"])
    head = ['time','ndc1','ndc2','ndc3','Trade_Partner_Name',
    'Distribution_Center_State','NDC','Distribution_Center_ID_(IC)',
    'Distribution_Center_Zip','Eff_Inv_(EU)','Eff_Inv_(PU)',
    'Qty_Ord_(EU)','Qty_Ord_(PU)']

    data = pd.DataFrame(M, columns=head)
    data["time"] = pd.to_datetime(data["time"], format='%Y%m%d', errors='coerce')

    if writ: # h5 allows your variable to be external
        dt = pd.HDFStore("drugdata.h5") # don't need to import/export!
        dt['dat'] = data
    return(data)

def test_hd5():
    dt = pd.HDFStore("drugdata.h5")
    print('here')
    #header = dt['dat'].columns.tolist()
    #for col in header:
        #print(dt['dat'][col].head())
    data = dt['dat']
    return(data)

if __name__ == '__main__':
    #data = impmat()
    #print(data)
    df = test_hd5()
    #df['year'], df['month'] = df['time'].dt.year, df['time'].dt.month
    
    TotalSales = df.groupby("NDC")['Qty_Ord_(EU)'].sum()
    print(TotalSales.nlargest(5))
    #TotalSales = np.array(TotalSales)
    #NDCs = df.groupby("NDC")
    #TotalSales.sort()
    #TotalSales = TotalSales[::-1]
    #print(TotalSales)
    #df2 = pd.DataFrame(TotalSales, index=NDCs, columns=['sales'])
    #df2.sort_values(by=['sales'])
    #print(df2)
    
    
    
    
    
    #print(df['time'])
    #NDC = 4 has the highest total sales
    #ndc4 = df.loc[df["NDC"]==23]
    #print("NDC4: ")
    #print(ndc4)
    #ndc4TotalSales = ndc4.groupby('time')['Qty_Ord_(EU)'].sum()
    #print("NDC4 Total Sales: ")
    #print(ndc4TotalSales)
    #ndc4Times = ndc4.time.unique()
    #print("NDC4 Times: ")
    #print(ndc4Times)
    #ndc4Times = np.array(ndc4Times)
    #ndc4Times.sort()
    #print("NDC4 Times (sorted): ")
    #print(ndc4Times)
    #ndc4TotalSales = np.array(ndc4TotalSales)
    #plt.plot(ndc4Times, ndc4TotalSales, 'bo')
    #plt.show()
    #X = [[ndc4Times], [ndc4TotalSales]]
    #X = np.empty([530,2], dtype='datetime64')
    #autocorrelation_plot(ndc4TotalSales)
    #plt.show()
    #print(X)
    #d = {'times': ndc4Times, 'sales': ndc4TotalSales}
    #df2 = pd.DataFrame(index=ndc4Times, data=ndc4TotalSales, columns=['sales'])
    #print(df2)
    
    #df2.plot()
    #plt.show()