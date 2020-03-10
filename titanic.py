#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
plt.interactive(True)

os.chdir('/Users/ankitmahna/Documents/Kaggle/Titanic/titanic/')

pd.set_option('display.expand_frame_repr', False)
train = pd.read_csv('train.csv')
oos_test = pd.read_csv('test.csv')

# Descriptives
train.info()
train.columns
oos_test.columns

# Null value imputation
# train data
train['Age'] = train['Age'].fillna(train['Age'].mean(skipna = True))
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])
train.info()

# oos_test data
oos_test['Age'] = oos_test['Age'].fillna(oos_test['Age'].mean(skipna = True))
oos_test['Fare'] = oos_test['Fare'].fillna(oos_test['Fare'].mean(skipna = True))

# Creating dummy variables
cols = ['Name','Ticket','Cabin','PassengerId']
train = train.drop(columns = cols)
oos_test = oos_test.drop(columns = cols)

obj_var = ['Pclass','Sex','Embarked']
train = pd.get_dummies(data = train, columns = obj_var)
oos_test = pd.get_dummies(data = oos_test, columns = obj_var)

# train test split
x_train, x_test, y_train, y_test = train_test_split(train.drop(columns = 'Survived'), train['Survived'], random_state=0)

############## Training LogisticRegression model
model = LogisticRegression(random_state=0, max_iter = 500)
model.fit(x_train, y_train)

# Model accuracy and validation
# train predictions
y_train_pred = model.predict(x_train)
# test predictions
y_test_pred = model.predict(x_test)
# train accuracy
model.score(x_test, y_test)
# test accuracy
model.score(x_test, y_test)
# confusion matrix
confusion_matrix(y_train, y_train_pred)
confusion_matrix(y_test, y_test_pred)
y_train.value_counts()
pd.Series(y_train_pred).value_counts()
# classification report
print(classification_report(y_train, y_train_pred, target_names = ['0: Not Survived', '1: Survived']))
print(classification_report(y_test, y_test_pred, target_names = ['0: Not Survived', '1: Survived']))


############### RandomForestCLassifier
rf = RandomForestClassifier(min_samples_split = 5, min_samples_leaf = 2, random_state = 0, bootstrap = True)
para_rf = {'max_depth':[3,4,5,6], 'max_features':[5,6,7], 'max_samples':[100,200,300]}
model_rf = GridSearchCV(estimator = rf , param_grid = para_rf, n_jobs = 2, cv = 3, return_train_score = True)
model_rf.fit(x_train, y_train)

# Model validation
model_rf.best_estimator_
# Train score
model_rf.score(x_train, y_train)
# test score
model_rf.score(x_test, y_test)
# predictions
y_train_pred_rf = model_rf.predict(x_train)
y_test_pred_rf = model_rf.predict(x_test)
confusion_matrix(y_train, y_train_pred_rf)
confusion_matrix(y_test, y_test_pred_rf)
print(classification_report(y_train, y_train_pred_rf, target_names = ['0: Not Survived', '1: Survived']))
print(classification_report(y_test, y_test_pred_rf, target_names = ['0: Not Survived', '1: Survived']))

oos_test_pred = pd.DataFrame(model_rf.predict(oos_test), columns = ['Survived'])
oos_test.merge(oos_test_pred, left_index=True, right_index=True)
oos_test_pred.to_csv('gender_submission.csv')