# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:23:41 2021

@author: Aakash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data = pd.read_csv(r'C:/Users/Aakash/Desktop/AAKASH/Coding Stuff/Full Data Science/Lesson 2/Assginments/Linear Regression Hand OUt/emp.csv')

'''
Exploratory data anlyasis and preprocessing
'''
print(data.head())
print(data.describe())

data.columns = ['salary','churn']

plt.hist(data['salary'], bins = 12)
plt.boxplot(data['salary'])

plt.hist(data['churn'], bins = 12)
plt.boxplot(data['churn'])

np.corrcoef(data.salary,data.churn)

def error_rmse(data, column, predict):
    sub = data[column] - predict
    sqr = sub*sub
    mse = np.mean(sqr)
    return np.sqrt(mse)

error_list = []

'''
Model 1
'''

plt.scatter(x = data['salary'], y = data['churn'], color = 'blue')
plt.xlabel('salary')
plt.ylabel('churn')
plt.show()

model_1 = smf.ols('churn ~ salary',data = data).fit()
print(model_1.summary())

prediction_1 = model_1.predict(pd.DataFrame(data['salary']))
 
error_1 = error_rmse(data, 'churn',prediction_1)
error_list.append(['base',error_1])
print(error_1)

'''
Model 2 
'''
plt.scatter(x = np.log(data['salary']), y = (data['churn']), color = 'blue')
plt.xlabel('salary')
plt.ylabel('churn')
plt.show()

model_2 = smf.ols('churn ~ np.log(salary)',data = data).fit()
print(model_2.summary())

prediction_2 = model_2.predict(pd.DataFrame(data['salary']))
 
error_2 = error_rmse(data, 'churn',prediction_2)
error_list.append(['base',error_2])
print(error_2)

'''
Model 3
'''

plt.scatter(x = (data['salary']), y = np.log(data['churn']), color = 'blue')
plt.xlabel('salary')
plt.ylabel('churn')
plt.show()

model_3 = smf.ols('np.log(churn) ~ (salary)',data = data).fit()
print(model_3.summary())

prediction_3 = model_3.predict(pd.DataFrame(data['salary']))
 
error_3 = error_rmse(data, 'churn',prediction_3)
error_list.append(['base',error_3])
print(error_3)

'''
Model 4
'''

plt.scatter(x = (data['salary']), y = np.square(data['churn']), color = 'blue')
plt.xlabel('salary')
plt.ylabel('churn')
plt.show()

model_4 = smf.ols('np.log(churn) ~ (salary)',data = data).fit()
print(model_4.summary())

prediction_4 = model_4.predict(pd.DataFrame(data['salary']))
 
error_4 = error_rmse(data, 'churn',prediction_4)
error_list.append(['base',error_4])
print(error_4)

'''
Model 5
'''

plt.scatter(x = np.square(np.log(data['salary'])), y = np.square(data['churn']), color = 'blue')
plt.xlabel('salary')
plt.ylabel('churn')
plt.show()

model_5 = smf.ols('np.square(churn) ~ np.log(salary)',data = data).fit()
print(model_5.summary())

prediction_5 = model_5.predict(pd.DataFrame(data['salary']))
 
error_5 = error_rmse(data, 'churn',prediction_5)
error_list.append(['base',error_5])
print(error_5)

'''
Final Model
'''

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size = 0.2)

final_model = smf.ols('churn ~ np.log(salary)',data = train_data).fit()

summary_final = final_model.summary()
print(summary_final)

final_train_prediction = final_model.predict(pd.DataFrame(train_data))
final_test_prediction = final_model.predict(pd.DataFrame(test_data))

final_train_error = error_rmse(data,'churn',final_train_prediction)
final_test_error = error_rmse(data,'churn',final_test_prediction)

final_train_error
final_test_error
    