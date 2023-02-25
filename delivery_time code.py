# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:43:53 2021

@author: Aakash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data = pd.read_csv(r'time.csv')

'''
Exploratory data anlyasis and preprocessing
'''
print(data.head())
print(data.describe())

data.columns = ['delivery','sorting']

plt.hist(data['delivery'], bins = 12)
plt.boxplot(data['delivery'])

plt.hist(data['sorting'], bins = 12)
plt.boxplot(data['sorting'])

np.corrcoef(data.delivery,data.sorting)

def error_rmse(data, column, predict):
    sub = data[column] - predict
    sqr = sub*sub
    mse = np.mean(sqr)
    return np.sqrt(mse)

error_list = []

'''
Model 1: Base Model
'''
 
plt.scatter(x=data['sorting'], y=data['delivery'], color='blue') 
plt.xlabel('sorting')
plt.ylabel('delivary')
plt.show()

model = smf.ols('delivery ~ sorting',data = data).fit()
print(model.summary())

prediction = model.predict(pd.DataFrame(data['sorting']))
 
error = error_rmse(data, 'delivery',prediction)
error_list.append(['base',error])
print(error)

'''
Model 2: Log transformation
'''

plt.scatter(x = np.log(data['sorting']), y = data['delivery'], color = 'blue')
plt.xlabel('sorting')
plt.ylabel('delivery')
plt.show()

model_log = smf.ols('delivery ~ np.log(sorting)',data = data).fit()
print(model_log.summary())

prediction_log = model_log.predict(pd.DataFrame(data['sorting']))
 
error_log = error_rmse(data, 'delivery',prediction_log)
error_list.append(['log',error_log])
print(error_log)

'''
Model 3
'''
plt.scatter(x = data['sorting'], y = np.log(data['delivery']), color = 'blue')
plt.xlabel('sorting')
plt.ylabel('delivery')
plt.show()

model_3 = smf.ols('np.log(delivery) ~ sorting',data = data).fit()
print(model_3.summary())

prediction_3 = model_3.predict(pd.DataFrame(data['sorting']))
 
error_3 = error_rmse(data, 'delivery',prediction_3)
error_list.append(['3',error_3])
print(error_3)

'''
Model_4
'''
plt.scatter(x = np.square(data['sorting']), y = np.log(data['delivery']), color = 'blue')
plt.xlabel('sorting')
plt.ylabel('delivery')
plt.show()

model_4 = smf.ols('np.log(delivery) ~ np.square(sorting)',data = data).fit()
print(model_4.summary())

prediction_4 = model_4.predict(pd.DataFrame(data['sorting']))
 
error_4 = error_rmse(data, 'delivery',prediction_4)
error_list.append(['4',error_4])
print(error_4)

'''
Picking a model
'''

error_data = pd.DataFrame(error_list)
error_data.columns = ['type','error']
error_data

'''
Final Model
'''

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size = 0.2)

final_model = smf.ols('delivery ~ np.log(sorting)',data = train_data).fit()

summary_final = final_model.summary()
print(summary_final)

final_train_prediction = final_model.predict(pd.DataFrame(train_data))
final_test_prediction = final_model.predict(pd.DataFrame(test_data))

final_train_error = error_rmse(data,'delivery',final_train_prediction)
final_test_error = error_rmse(data,'delivery',final_test_prediction)

final_train_error
final_test_error
