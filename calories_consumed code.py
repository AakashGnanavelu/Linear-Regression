# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:00:45 2021

@author: Aakash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def error_rmse(data, column, predict):
    sub = data[column] - predict
    sqr = sub*sub
    mse = np.mean(sqr)
    return np.sqrt(mse)

data = pd.read_csv("cc.csv")
print(data.head())
print(data.describe())

data.columns = ['weight','cal']

plt.hist(data.weight) #histogram
plt.boxplot(data.weight) #boxplot

# MPG
plt.hist(data.cal) #histogram
plt.boxplot(data.cal) #boxplot

# correlation
np.corrcoef(data.weight, data.cal) 

'''
Model 1: Base Model
'''
# Scatter plot
plt.scatter(x=data['cal'], y=data['weight'], color='green') 
plt.xlabel('cal')
plt.ylabel('weight')
plt.show()

model = smf.ols('weight ~ cal', data = data).fit()

summary = model.summary()
print(summary)

prediction = model.predict(pd.DataFrame(data['cal']))

model_error = error_rmse(data,'weight',prediction)
model_error

'''
Model 2 : Log Transformation
'''
plt.scatter(x=np.log(data['cal']), y=data['weight'], color='green') 
plt.xlabel('cal with log')
plt.ylabel('weight')
plt.show()

model_log = smf.ols('weight ~ np.log(cal)', data = data).fit()

summary_log = model_log.summary()
print(summary_log)

prediction_log = model_log.predict(pd.DataFrame(data['cal']))

model_error_log = error_rmse(data,'weight',prediction_log)
model_error_log
'''
Model 3 : Square Transformation
'''

def sq(num):
    return num*num

plt.scatter(x= data['cal'], y = sq(data['weight']), color='green') 
plt.xlabel('cal')
plt.ylabel('weight with with sqrt')
plt.show()

model_sq = smf.ols('sq(weight) ~ cal', data = data).fit()

summary_sq = model_sq.summary()
print(summary_sq)

prediction_sq = model_sq.predict(pd.DataFrame(data['cal']))

model_error_sq = error_rmse(data,'weight',prediction_sq)
model_error_sq

'''
Model 4 : Sqrt Transformation

'''
plt.scatter(x= data['cal'], y = np.sqrt(data['weight']), color='green') 
plt.xlabel('cal')
plt.ylabel('weight with with sqrt')
plt.show()

model_sqrt = smf.ols('np.sqrt(weight) ~ cal', data = data).fit()

summary_sqrt = model_sqrt.summary()
print(summary_sqrt)

prediction_sqrt = model_sqrt.predict(pd.DataFrame(data['cal']))

model_error_sqrt = error_rmse(data,'weight',prediction_sqrt)
model_error_sqrt

'''
Model 5.

'''
plt.scatter(x= np.log(data['cal']), y = sq(data['weight']), color='green') 
plt.xlabel('cal with log')
plt.ylabel('weight with with log')
plt.show()

model_sq_log = smf.ols('np.log(weight) ~ sq(cal)', data = data).fit()

summary_sq_log = model_sq_log.summary()
print(summary_sqrt)

prediction_sq_log = model_sq_log.predict(pd.DataFrame(data['cal']))

model_error_sq_log = error_rmse(data,'weight',prediction_sq_log)
model_error_sq_log


'''
Model 5.

'''
def cube(num):
    return num*num*num

plt.scatter(x= cube(data['cal']), y = data['weight'], color='green') 
plt.xlabel('cal ^2')
plt.ylabel('weight ^3')
plt.show()

model_exp = smf.ols('np.sqrt(weight) ~ cube(cal)', data = data).fit()

summary_exp = model_exp.summary()
print(summary_sqrt)

prediction_exp = model_exp.predict(pd.DataFrame(data['cal']))

model_error_exp = error_rmse(data,'weight',prediction_exp)
model_error_exp
'''
Comparing Models
'''
model_error # best one
model_error_exp
model_error_log
model_error_sq
model_error_sq_log
model_error_sqrt

'''
Final Model
'''

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size = 0.2)

final_model= smf.ols('weight ~ cal', data =train_data).fit()

summary_final = final_model.summary()
print(summary_final)

final_train_prediction = final_model.predict(pd.DataFrame(train_data))
final_test_prediction = final_model.predict(pd.DataFrame(test_data))

final_train_error = error_rmse(data,'weight',final_train_prediction)
final_test_error = error_rmse(data,'weight',final_test_prediction)

final_train_error
final_test_error


