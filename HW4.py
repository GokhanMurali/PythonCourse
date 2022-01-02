import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#read from csv file
df = pd.read_csv('immSurvey.csv')

#construct features and target matrices
X = df['text']
Y = df['sentiment']

#counts words
vec = CountVectorizer()
X_processed = vec.fit_transform(X)
#pd.DataFrame(X_processed.toarray(), columns=vec.get_feature_names())

#split the data as train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_processed, Y, random_state=0, train_size=0.5, test_size=0.5)

#Support Vector Regressor(SVR)
regressor = SVR(kernel = 'poly')
regressor.fit(Xtrain, Ytrain)
y_pred = regressor.predict(Xtest)
#calculated MSE to measure model performance
MSE = mean_squared_error(Ytest, y_pred)
MSE

#For SVR optimization, I will use GridSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np

grid_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma' : ['scale', 'auto']}

grid = GridSearchCV(SVR(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain,Ytrain)
grid.best_params_

#run SVR with best parameters
regressor_optimized = SVR(kernel = 'rbf', gamma = 'scale')
regressor_optimized.fit(Xtrain, Ytrain)
y_pred_optimized = regressor_optimized.predict(Xtest)
MSE_optimized = mean_squared_error(Ytest, y_pred_optimized)
MSE_optimized

from sklearn.ensemble import RandomForestRegressor
#Random Forest Regressor(RFR)
regressor_RFR = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_RFR.fit(Xtrain, Ytrain)
y_pred_RFR = regressor_RFR.predict(Xtest)
#calculated MSE to measure model performance
MSE_RFR = mean_squared_error(Ytest, y_pred_RFR)
MSE_RFR

#For RFR optimization, I will use GridSearchCV
grid_params = {'criterion' : ['mse', 'mae'],
               'max_features' : ['auto', 'sqrt', 'log2'],
               'bootstrap' :[True, False]}

max_features : {"auto", "sqrt", "log2"}

grid = GridSearchCV(RandomForestRegressor(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain,Ytrain)
grid.best_params_

#run RFR with best parameters
regressor_RFR_optimized = RandomForestRegressor(n_estimators = 100, random_state = 0, criterion = 'mae', max_features = 'sqrt', bootstrap = False)
regressor_RFR_optimized.fit(Xtrain, Ytrain)
y_pred_RFR_optimized = regressor_RFR_optimized.predict(Xtest)
#calculated MSE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred_RFR_optimized)
MSE_RFR_optimized

from sklearn.feature_extraction.text import TfidfVectorizer
#weights the word counts by a measure of how often they appear in the documents
vec_Tfid = TfidfVectorizer()
X_processed_TFid = vec_Tfid.fit_transform(X)
#pd.DataFrame(X_processed_TFid.toarray(), columns=vec_Tfid.get_feature_names())

#split the data as train and test. notice X_processed changed (X_processed_TFid)
Xtrain_TFid, Xtest_TFid, Ytrain_TFid, Ytest_TFid = train_test_split(X_processed_TFid, Y,
random_state=0, train_size=0.5, test_size=0.5) #split the data as train and test

#For SVR optimization, I will use GridSearchCV
grid_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma' : ['scale', 'auto']}

grid = GridSearchCV(SVR(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain_TFid,Ytrain_TFid)
grid.best_params_

#run SVR with best parameters
regressor_TFid_optimized = SVR(kernel = 'linear', gamma = 'scale')
regressor_TFid_optimized.fit(Xtrain_TFid, Ytrain_TFid)
y_pred_TFid_optimized = regressor_TFid_optimized.predict(Xtest_TFid)
#calculated MSE to measure model performance
MSE_TFid_optimized = mean_squared_error(Ytest_TFid, y_pred_TFid_optimized)
MSE_TFid_optimized

#For RFR optimization, I will use GridSearchCV
grid_params = {'criterion' : ['mse', 'mae'],
               'max_features' : ['auto', 'sqrt', 'log2'],
               'bootstrap' :[True, False]}

max_features : {"auto", "sqrt", "log2"}

grid = GridSearchCV(RandomForestRegressor(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain_TFid,Ytrain_TFid)
grid.best_params_

#run RFR with best parameters
regressor_TFid_RFR_optimized = RandomForestRegressor(n_estimators = 100, random_state = 0, criterion = 'mse', max_features = 'sqrt', bootstrap = False)
regressor_TFid_RFR_optimized.fit(Xtrain_TFid, Ytrain_TFid)
y_pred_TFid_RFR_optimized = regressor_TFid_RFR_optimized.predict(Xtest_TFid)
#calculated MSE to measure model performance
MSE_TFid_RFR_optimized = mean_squared_error(Ytest_TFid, y_pred_TFid_RFR_optimized)
MSE_TFid_RFR_optimized

#use bigrams instead of unigrams for CountVectorizer
vec_bigrams = CountVectorizer(ngram_range=(2,2))
X_processed_bigrams = vec_bigrams.fit_transform(X)
#pd.DataFrame(X_processed_bigrams.toarray(), columns=vec_bigrams.get_feature_names())

#split the data as train and test. notice X_processed changed (X_processed_bigrams)
Xtrain_bigrams, Xtest_bigrams, Ytrain_bigrams, Ytest_bigrams = train_test_split(X_processed_bigrams, Y,
random_state=0, train_size=0.5, test_size=0.5) #split the data as train and test

#For SVR optimization, I will use GridSearchCV
grid_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma' : ['scale', 'auto']}

grid = GridSearchCV(SVR(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain_bigrams, Ytrain_bigrams)
grid.best_params_

#run SVR with best parameters
regressor_bigrams_optimized = SVR(kernel = 'rbf', gamma = 'scale')
regressor_bigrams_optimized.fit(Xtrain_bigrams, Ytrain_bigrams)
y_pred_bigrams_optimized = regressor_bigrams_optimized.predict(Xtest_bigrams)
#calculated MSE to measure model performance
MSE_bigrams_optimized = mean_squared_error(Ytest_bigrams, y_pred_bigrams_optimized)
MSE_bigrams_optimized

#For RFR optimization, I will use GridSearchCV
grid_params = {'criterion' : ['mse', 'mae'],
               'max_features' : ['auto', 'sqrt', 'log2'],
               'bootstrap' :[True, False]}

max_features : {"auto", "sqrt", "log2"}

grid = GridSearchCV(RandomForestRegressor(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain_bigrams,Ytrain_bigrams)
grid.best_params_

#run RFR with best parameters
regressor_bigrams_RFR_optimized = RandomForestRegressor(n_estimators = 100, random_state = 0, criterion = 'mae', max_features = 'log2', bootstrap = True)
regressor_bigrams_RFR_optimized.fit(Xtrain_bigrams, Ytrain_bigrams)
y_pred_bigrams_RFR_optimized = regressor_bigrams_RFR_optimized.predict(Xtest_bigrams)
#calculated MSE to measure model performance
MSE_bigrams_RFR_optimized = mean_squared_error(Ytest_bigrams, y_pred_bigrams_RFR_optimized)
MSE_bigrams_RFR_optimized

#use bigrams instead of unigrams for TfidfVectorizer
vec_Tfid_bigrams = TfidfVectorizer(ngram_range=(2,2))
X_processed_TFid_bigrams = vec_Tfid_bigrams.fit_transform(X)
#pd.DataFrame(X_processed_TFid.toarray(), columns=vec.get_feature_names())

#split the data as train and test. notice X_processed changed (X_processed_TFid_bigrams)
Xtrain_TFid_bigrams, Xtest_TFid_bigrams, Ytrain_TFid_bigrams, Ytest_TFid_bigrams = train_test_split(X_processed_TFid_bigrams, Y,
random_state=0, train_size=0.5, test_size=0.5) #split the data as train and test

#For SVR optimization, I will use GridSearchCV
grid_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma' : ['scale', 'auto']}

grid = GridSearchCV(SVR(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain_TFid_bigrams, Ytrain_TFid_bigrams)
grid.best_params_

#run SVR with best parameters
regressor_TFid_bigrams_optimized = SVR(kernel = 'rbf', gamma = 'scale')
regressor_TFid_bigrams_optimized.fit(Xtrain_TFid_bigrams, Ytrain_TFid_bigrams)
y_pred_TFid_bigrams_optimized = regressor_TFid_bigrams_optimized.predict(Xtest_TFid_bigrams)
#calculated MSE to measure model performance
MSE_TFid_bigrams_optimized = mean_squared_error(Ytest_TFid_bigrams, y_pred_TFid_bigrams_optimized)
MSE_TFid_bigrams_optimized

#For RFR optimization, I will use GridSearchCV
grid_params = {'criterion' : ['mse', 'mae'],
               'max_features' : ['auto', 'sqrt', 'log2'],
               'bootstrap' :[True, False]}

max_features : {"auto", "sqrt", "log2"}

grid = GridSearchCV(RandomForestRegressor(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain_TFid_bigrams,Ytrain_TFid_bigrams)
grid.best_params_

#run RFR with best parameters
regressor_TFid_bigrams_RFR_optimized = RandomForestRegressor(n_estimators = 100, random_state = 0, criterion = 'mae', max_features = 'log2', bootstrap = True)
regressor_TFid_bigrams_RFR_optimized.fit(Xtrain_TFid_bigrams, Ytrain_TFid_bigrams)
y_pred_TFid_bigrams_RFR_optimized = regressor_TFid_bigrams_RFR_optimized.predict(Xtest_TFid_bigrams)
#calculated MSE to measure model performance
MSE_TFid_bigrams_RFR_optimized = mean_squared_error(Ytest_TFid_bigrams, y_pred_TFid_bigrams_RFR_optimized)
MSE_TFid_bigrams_RFR_optimized
