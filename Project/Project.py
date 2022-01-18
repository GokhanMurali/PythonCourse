import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#read from csv file
df = pd.read_csv('insurance_data.csv', delimiter=';')
df['bmi'] = df['bmi'].replace(',','.', regex=True).astype(float)
df['medical cost'] = df['medical cost'].replace(',','.', regex=True).astype(float)


#construct features and target matrices
X = df.iloc[:,0:5]
Y = df['medical cost']

#for bmi column there are missing data I replaced missing values with most_frequent items
X_processed = SimpleImputer(strategy='most_frequent', missing_values=99)
X_processed = X_processed.fit(X[['bmi']])
X['bmi'] = X_processed.transform(X[['bmi']])

# sex and smoker data are categorical, thefore I will apply one-hot encoding
# creating instance of one-hot-encoder
X_encoded = OneHotEncoder(sparse=True, handle_unknown='ignore')
# passing sex and smoker columns
X_encoded = pd.DataFrame(X_encoded.fit_transform(X[['sex', 'smoker']]).toarray())
# merge encoded columns with other columns
X_encoded = X_encoded.join(X['age'])
X_encoded = X_encoded.join(X['bmi'])
X_encoded = X_encoded.join(X['children'])

#split the data as train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_encoded, Y, random_state=0,
train_size=0.8, test_size=0.2)

#LinearRegression
model = LinearRegression(fit_intercept=True) #storing of hyperparameter values
model.fit(Xtrain, Ytrain)
y_pred_LR = model.predict(Xtest)
#calculated MSE, MAE to measure model performance
MSE_LR = mean_squared_error(Ytest, y_pred_LR)
MAE_LR = mean_absolute_error(Ytest, y_pred_LR)
MSE_LR,MAE_LR

#For LR optimization, I will use GridSearchCV
grid_params = {'fit_intercept': [True, False],
               'normalize' : [True, False],
               'copy_X' : [True, False]}

grid = GridSearchCV(LinearRegression(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain,Ytrain)
grid.best_params_

#run LinearRegression with optimized parameters
model_optimized = LinearRegression(fit_intercept=True, normalize=True,copy_X=True) #storing of hyperparameter values
model_optimized.fit(Xtrain, Ytrain)
y_pred_LR_optimized = model_optimized.predict(Xtest)
#calculated MSE, MAE to measure model performance
MSE_LR_optimized = mean_squared_error(Ytest, y_pred_LR_optimized)
MAE_LR_optimized = mean_absolute_error(Ytest, y_pred_LR_optimized)
MSE_LR_optimized,MAE_LR_optimized

#Support Vector Regressor(SVR)
regressor = SVR(kernel = 'poly')
regressor.fit(Xtrain, Ytrain)
y_pred = regressor.predict(Xtest)
#calculated MSE, MAE to measure model performance
MSE = mean_squared_error(Ytest, y_pred)
MAE = mean_absolute_error(Ytest, y_pred)
MSE,MAE

#For SVR optimization, I will use GridSearchCV
grid_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma' : ['scale', 'auto']}

grid = GridSearchCV(SVR(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain,Ytrain)
grid.best_params_

#run SVR with best parameters
regressor_optimized = SVR(kernel = 'poly', gamma = 'auto')
regressor_optimized.fit(Xtrain, Ytrain)
y_pred_optimized = regressor_optimized.predict(Xtest)
MSE_optimized = mean_squared_error(Ytest, y_pred_optimized)
MAE_optimized = mean_absolute_error(Ytest, y_pred_optimized)
MSE_optimized,MAE_optimized

#Random Forest Regressor(RFR)
regressor_RFR = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_RFR.fit(Xtrain, Ytrain)
y_pred_RFR = regressor_RFR.predict(Xtest)
#calculated MSE,MAE to measure model performance
MSE_RFR = mean_squared_error(Ytest, y_pred_RFR)
MAE_RFR = mean_absolute_error(Ytest, y_pred_RFR)
MSE_RFR,MAE_RFR

#For RFR optimization, I will use GridSearchCV
grid_params = {'criterion' : ['mse', 'mae'],
               'max_features' : ['auto', 'sqrt', 'log2'],
               'bootstrap' :[True, False]}

max_features : {"auto", "sqrt", "log2"}

grid = GridSearchCV(RandomForestRegressor(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain,Ytrain)
grid.best_params_

#run RFR with best parameters
regressor_RFR_optimized = RandomForestRegressor(n_estimators = 100, random_state = 0, criterion = 'mae', max_features = 'sqrt', bootstrap = True)
regressor_RFR_optimized.fit(Xtrain, Ytrain)
y_pred_RFR_optimized = regressor_RFR_optimized.predict(Xtest)
#calculated MSE,MAE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred_RFR_optimized)
MAE_RFR_optimized = mean_absolute_error(Ytest, y_pred_RFR_optimized)
MSE_RFR_optimized,MAE_RFR_optimized

#regression between sex and medical cost
regressor_RFR_optimized.fit(np.reshape(np.array(Xtrain[0]), (-1, 1)), Ytrain)
y_pred1 = regressor_RFR_optimized.predict(np.reshape(np.array(Xtest[0]), (-1, 1)))
#calculated MSE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred1)
MAE_RFR_optimized = mean_absolute_error(Ytest, y_pred1)
MSE_RFR_optimized,MAE_RFR_optimized

#regression between smoking and medical cost
regressor_RFR_optimized.fit(np.reshape(np.array(Xtrain[2]), (-1, 1)), Ytrain)
y_pred1 = regressor_RFR_optimized.predict(np.reshape(np.array(Xtest[2]), (-1, 1)))
#calculated MSE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred1)
MAE_RFR_optimized = mean_absolute_error(Ytest, y_pred1)
MSE_RFR_optimized,MAE_RFR_optimized

#regression between bmi and medical cost
regressor_RFR_optimized.fit(np.reshape(np.array(Xtrain['bmi']), (-1, 1)), Ytrain)
y_pred1 = regressor_RFR_optimized.predict(np.reshape(np.array(Xtest['bmi']), (-1, 1)))
#calculated MSE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred1)
MAE_RFR_optimized = mean_absolute_error(Ytest, y_pred1)
MSE_RFR_optimized,MAE_RFR_optimized

#regression between age and medical cost
regressor_RFR_optimized.fit(np.reshape(np.array(Xtrain['age']), (-1, 1)), Ytrain)
y_pred1 = regressor_RFR_optimized.predict(np.reshape(np.array(Xtest['age']), (-1, 1)))
#calculated MSE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred1)
MAE_RFR_optimized = mean_absolute_error(Ytest, y_pred1)
MSE_RFR_optimized,MAE_RFR_optimized

#regression between children and medical cost
regressor_RFR_optimized.fit(np.reshape(np.array(Xtrain['children']), (-1, 1)), Ytrain)
y_pred1 = regressor_RFR_optimized.predict(np.reshape(np.array(Xtest['children']), (-1, 1)))
#calculated MSE to measure model performance
MSE_RFR_optimized = mean_squared_error(Ytest, y_pred1)
MAE_RFR_optimized = mean_absolute_error(Ytest, y_pred1)
MSE_RFR_optimized,MAE_RFR_optimized
