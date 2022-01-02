import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

#read from csv file
df = pd.read_csv('cses4_cut.csv')
#construct features matrix, my features are D2003: education and age
X = df[['D2003', 'D2010', 'age']]
#target matrix
Y = df['voted']

#for D2003 and D2010 columns replaced missing values with most_frequent items
X_processed = SimpleImputer(strategy='most_frequent', missing_values=99)
X_processed = X_processed.fit(X[['D2003']])
X['D2003'] = X_processed.transform(X[['D2003']])
X_processed = X_processed.fit(X[['D2010']])
X['D2010'] = X_processed.transform(X[['D2010']])

# education is categorical, thefore I will apply one-hot encoding
# creating instance of one-hot-encoder
X_encoded = OneHotEncoder(sparse=True, handle_unknown='ignore')
# passing bridge-types-cat column
X_encoded = pd.DataFrame(X_encoded.fit_transform(X[['D2003', 'D2010']]).toarray())
# merge encoded columns with age column
X_encoded = X_encoded.join(X['age'])

#split the data as train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_encoded, Y,
random_state=1)

#First, I will use Gaussian naive Bayes model for classification
model = GaussianNB() #used Gaussian naive Bayes model for classification
model.fit(Xtrain, Ytrain) #fit the model
Y_model = model.predict(Xtest) #predict y values for x test

accuracy_score(Ytest, Y_model) #calculates accuracy score of model

#where did we go wrong? Confusion matrix shows frequency of misclassification
import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(Ytest, Y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

#Second, I will use KNeighborsClassifier model for classification

model2 = KNeighborsClassifier(n_neighbors=1)
model2.fit(Xtrain, Ytrain)

Y_KN_model = model2.predict(Xtest)
accuracy_score(Ytest, Y_KN_model)

#For model optimization, I will use GridSearchCV

from sklearn.model_selection import GridSearchCV
import numpy as np

grid_params = {'n_neighbors': np.arange(20),
               'weights': ['uniform', 'distance'],
               'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv = 7, n_jobs = -1)

grid.fit(Xtrain,Ytrain)
grid.best_params_

#Finally, I will use KNeighborsClassifier model with best parameters

model = KNeighborsClassifier(n_neighbors=17, weights='uniform', algorithm='brute')
model.fit(Xtrain, Ytrain)

Y_KN_model2 = model.predict(Xtest)
accuracy_score(Ytest, Y_KN_model2)

#where did we go wrong? Confusion matrix shows frequency of misclassification
mat = confusion_matrix(Ytest, Y_KN_model2)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
