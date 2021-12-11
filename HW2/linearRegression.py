import pandas as pd
from pandas import read_csv
import numpy as np
from scipy import stats
import scipy

def linearRegression(data_file):
    df = read_csv(data_file, delimiter = ',')
    df2 = df.dropna()
    data_no_missing = np.array(df2.values, dtype="float64")
    rows, columns = data_no_missing.shape

    X = np.ones((rows, columns), dtype="float64") #creates X matrix with ones
    for i in range(rows):
        for j in range(columns-1):
            X[i][j+1] = data_no_missing[:, j+1][i] #first column has ones, other columns has X values

    Y = np.ones((rows,1), dtype="float64") #creates Y matrix with ones
    for i in range(rows):
        Y[i] = data_no_missing[:, 0][i] #appends Y values to Y matrix

    beta = np.array(np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)),X.T),Y), dtype="float64") #finds beta values

    Y_predicted = np.ones((rows,1), dtype="float64")
    for i in range(rows):
        Y_predicted[i] = beta[0]
        for j in range(columns-1):
            Y_predicted[i] += beta[j+1]*X[i][j+1] #calculates predicted Y values

    error = np.ones((rows,1), dtype="float64")
    for i in range(rows):
        error[i] = Y[i] - Y_predicted[i] #calculates errors

    variance = np.matmul(error.T, error) / (rows - (columns - 1) - 1)  #calculates variance
    standard_deviations = np.sqrt(np.array(variance[0] * np.linalg.inv(np.matmul(X.T, X))).diagonal()) #calculates standard deviations

    confidence_intervals = np.ones((columns,2), dtype="float64")
    for i in range(len(df.columns)):
        confidence_intervals[i][0] = beta[i] - scipy.stats.t.ppf(q=.975,df=rows - (columns - 1) - 1) * standard_deviations[i]
        confidence_intervals[i][1] = beta[i] + scipy.stats.t.ppf(q=.975,df=rows - (columns - 1) - 1) * standard_deviations[i]

    #print(X)
    #print(Y)
    #print(X.T)
    #print(beta)
    #print(Y_predicted)
    #print(error)
    #print(variance)
    #print(standard_deviations)
    #print(confidence_intervals)
    return(beta, standard_deviations, confidence_intervals) #returns beta values, their standard deviation, their 95% confidence intervals
