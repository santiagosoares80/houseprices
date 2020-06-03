# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:20:26 2020

@author: santiago
"""


import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

X = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop('SalePrice', axis=1, inplace=True)
X.drop(['MSSubClass','MSZoning','Street', 'LotShape','LandContour','LandSlope',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
        'Electrical','Functional','GarageYrBlt','Fence','MoSold','YrSold','SaleType',
        'SaleCondition', 'Condition1', 'Condition2','BldgType','HouseStyle', 'FireplaceQu'], axis=1, inplace=True)
X_test.drop(['MSSubClass','MSZoning','Street', 'LotShape','LandContour','LandSlope',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
        'Electrical','Functional','GarageYrBlt','Fence','MoSold','YrSold','SaleType',
        'SaleCondition', 'Condition1', 'Condition2','BldgType','HouseStyle','FireplaceQu'], axis=1, inplace=True)

# Drop columns with too many NaN
X.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)
X_test.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1, inplace=True)

# Preprocessing data

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]

categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

numerical_transformer = Pipeline(steps=[
    ('imputer',  SimpleImputer(strategy='constant')),
    ('normalizer', StandardScaler())
    ])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
        ])

# # Linear regression
linreg_model = LinearRegression()

linreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', linreg_model)
    ])

linreg_scores = -1 * cross_val_score(linreg_pipeline, X, y, 
                        scoring='neg_mean_absolute_error', cv=5, 
                        n_jobs=-1)

print("Error for Linear Regression: ", linreg_scores)
print("Mean error for Linear Regression: ", linreg_scores.mean())

# # Random Forest Regression

rf_model = RandomForestRegressor(random_state=0)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
    ])

rf_scores = -1 * cross_val_score(rf_pipeline, X, y,
                                  scoring='neg_mean_absolute_error', cv=5,
                                  n_jobs=-1)

print("Error for Random Forest Regressor: ", rf_scores)
print("Mean error for Random Forest Regressor: ", rf_scores.mean())

# XGB Regressor

xgbr_model = XGBRegressor(objective='reg:squarederror', random_state=0)

xgbr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgbr_model)
    ])
                      
xgbr_scores = -1 * cross_val_score(xgbr_pipeline, X, y,
                                    scoring='neg_mean_absolute_error', cv=5,
                                    n_jobs=-1)

print("Error for XGBRegressor: ", xgbr_scores)
print("Mean error for XGBRegressor: ", xgbr_scores.mean())

# # SVM

svm_model = NuSVR()

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', svm_model)
    ])

svm_scores = -1 * cross_val_score(svm_pipeline, X, y,
                                  scoring='neg_mean_absolute_error', cv=5,
                                  n_jobs=-1)

print("Error for SVM: ", svm_scores)
print("Mean error for SVM: ", svm_scores.mean())

# Analyzing error vs. m (# of training examples)

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# model = XGBRegressor(objective='reg:squarederror', n_estimators=2000, learning_rate=0.04, max_depth=2, random_state=0)
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', model)
#     ])

# mses=pd.DataFrame(columns=['Training Set Size', 'MAE', 'Training/Validation'])

# for samples in range(10,1150,10):
#     # Shuffle training set and get the first "samples" lines
#     idx = np.random.permutation(X_train.index)
#     X_t = X_train.reindex(idx)
#     y_t = y_train.reindex(idx)
    
#     X_t = X_t.iloc[0:samples]
#     y_t = y_t.iloc[0:samples]

#     model_pipeline.fit(X_t, y_t)
    
#     # Preprocessing of validation data, get predictions
#     preds_train = model_pipeline.predict(X_t)
#     preds_valid = model_pipeline.predict(X_valid)
    
#     mse_preds_train = mean_squared_error(preds_train,y_t)
#     mse_preds_valid = mean_squared_error(preds_valid,y_valid)

#     mses = mses.append({'Training Set Size': samples, 'MSE': mse_preds_train, 'Training/Validation': "Training"}, ignore_index=True) # Your code here
#     mses = mses.append({'Training Set Size': samples, 'MSE': mse_preds_valid, 'Training/Validation': "Validation"}, ignore_index=True) # Your code here

# plt.figure(figsize=(14,8))
# sns.set_style('whitegrid')
# sns.lineplot(data=mses, x='Training Set Size', y='MSE', hue='Training/Validation')

# # Predicting for Kaggle Competition

param_grid = [{'model__learning_rate': [0.04, 0.05, 0.06], 'model__max_depth': [2, 3, 4], 'model__n_estimators': [2000]}]
search = GridSearchCV(xgbr_pipeline, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

search.fit(X,y)
print("Best params: ", search.best_params_)
print("Best score: ", -1 * search.best_score_)

preds = search.predict(X_test)

output = pd.DataFrame({'Id': X_test['Id'],
                        'SalePrice': preds})
output.to_csv('submission.csv', index=False)