# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:01:09 2018

@author: Jack

Read in term project data
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
    df['year'], df['month'] = df['time'].dt.year, df['time'].dt.month
    #print(df['time'])
    #NDC = 4 has the highest total sales
    ndc4 = df.loc[df["NDC"]==6]
    print("NDC4: ")
    print(ndc4)
    ndc4TotalSales = ndc4.groupby('time')['Qty_Ord_(EU)'].sum()
    print("NDC4 Total Sales: ")
    print(ndc4TotalSales)
    ndc4Times = ndc4.time.unique()
    print("NDC4 Times: ")
    print(ndc4Times)
    ndc4Times = np.array(ndc4Times)
    ndc4Times.sort()
    print("NDC4 Times (sorted): ")
    print(ndc4Times)
    ndc4TotalSales = np.array(ndc4TotalSales)
    plt.plot(ndc4Times, ndc4TotalSales, 'bo')
    plt.show()
    #X = [[ndc4Times], [ndc4TotalSales]]
    #X = np.empty([530,2], dtype='datetime64')
    autocorrelation_plot(ndc4TotalSales)
    plt.show()
    #print(X)
    #d = {'times': ndc4Times, 'sales': ndc4TotalSales}
    df2 = pd.DataFrame(index=ndc4Times, data=ndc4TotalSales, columns=['sales'])
    print(df2)
    
    df2.plot()
    plt.show()
    
    
    
    dftest = pd.DataFrame(index=ndc4Times[400:], data=ndc4TotalSales[400:], columns=['sales'])
    print(dftest)
    
    
    ########ARIMA#############
    
    """
    
    start = time.time()
    
    autocorrelation_plot(dftest)
    plt.show()
    
    model = ARIMA(dftest, order=(10,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())
    
    
    
    
    
    X = dftest.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    print("Train/test split")
    history = [x for x in train]
    predictions = list()
    print("About to FOR loop")
    for t in range(len(test)):
        model = ARIMA(history, order=(10,1,0))
        try: model_fit = model.fit(disp=0)
        except (ValueError, LinAlgError): pass
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    print("Out of FOR loop")
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()
    
    end = time.time()
    print("This took: ", end-start, " seconds")
    
    """
    X = ndc4Times
    Y1 = ndc4TotalSales
    
    start = time.time()
    #errors = []
    #J = np.arange(int(X.shape[0]/10))
    
    #for j in J:
    
    
    
    #suggested = int(X.shape[0]/10)
    #print(suggested)
    a=10
    #print(a)
    Y_past = [ Y1 ]
    for i in range(0,a) :
        Yi = Y_past[len(Y_past)-1]
        Yi = np.insert(Yi, 0, 0)
        Y_past.append(Yi[0:len(Y1)])
    Y_past = np.matrix(Y_past)
    #Y_past = np.transpose(Y_past)
    #print(Y1)
    #print("Y: ")
    #print(Y_past)
    
    #print("Y shape: ")
    #print(Y_past.shape)
    
    Y_past = np.delete(Y_past, 0, 0)
    #print("Y: ")
    #print(Y_past)
    Y = Y1
    #print("New shape of Y_past: ")
    Y_past = np.transpose(Y_past)
    #print(Y_past.shape)
    
    
    
    
    
    #print(X[0])
    Xts = (X - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    #print(Xts[0])
    Xts = Xts.reshape(Xts.shape[0], 1)
    #print(Xts.shape)
    
    IN = np.hstack((Xts, Y_past))
    #print("Input matrix: ")
    #print(IN)
    print("Input Shape: ")
    print(IN.shape)
    
    
    
    
    size = int(IN.shape[0]*0.66)
    #print(size)
    #X_train, X_test = Xts[0:size], Xts[size: len(Xts)]
    X_train, X_test = IN[0:size,:], IN[size:IN.shape[0],:]
    #[X_train, X_test] = np.vsplit(IN, size) ################
    Y_train, Y_test = Y[0:size], Y[size: len(Y)]
    X_train_64, X_test_64 = X[0:size], X[size: len(X)]
    Y_train_64, Y_test_64 = Y[0:size], Y[size: len(Y)]
    
    
    #X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    """
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("Y_train shape: ", Y_train.shape)
    print("Y_test shape: ", Y_test.shape)
    print("X_train type: ", X_train.dtype)
    print("X_test type: ", X_test.dtype)
    print("Y_train type: ", Y_train.dtype)
    print("Y_test type: ", Y_test.dtype)
    """
    regr = LassoCV()
    regr.fit(X_train, Y_train)
    
    pred_trained = []
    pred = []
    for x1 in X_train:
        #if j==0:
         #   x1 = x1.reshape(-1,1)
        yHat_trained = regr.predict(x1)
        pred_trained.append(yHat_trained)
    for x in X_test:
        #if j==0:
         #   x = x.reshape(-1,1)
        yHat = regr.predict(x)
        pred.append(yHat)
        
    #X_convert = X_train.reshape(X_train.shape[0]) 
    X_convert = datetime.utcfromtimestamp(X_train[0,0])
    #X_test_64 = datetime.utcfromtimestamp(X_test)
    #Y_test_64 = datetime.utcfromtimestamp(Y_test)
    #pred_64 = datetime.utcfromtimestamp(pred)
    X_convert = np.datetime64(X_convert)
    #X_test_64 = np.datetime64(X_test)
    #Y_test_64 = np.datetime64(Y_test)
    #pred_64 = np.datetime64(pred)
    
    #print(X_convert)
    
    plt.plot(X_train_64, Y_train, color='green')
    plt.plot(X_train_64, pred_trained, color='red')
    plt.plot(X_test_64, pred, color='orange')
    plt.plot(X_test_64, Y_test, color='blue')
    plt.show()
    
    error = mean_squared_error(Y_test, pred)
    print("Error: ", error)
    #errors.append(error)
    
    
                    
    #plt.plot(J, errors, color='red')
    #plt.show()
    
    end = time.time()
    print("This took: ", end-start, " seconds")
    
