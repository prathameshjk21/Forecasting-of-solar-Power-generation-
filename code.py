

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn import tree, svm ,linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


os.chdir("E:\data science\climate connect")

data1 = pd.read_csv("power_actual.csv")
data2 = pd.read_csv("weather_actuals.csv")

#################cleaning the data##################


data1 = data1.reset_index()

col = ["ghi" , "gti","Unnamed: 0","index"]

data1 = data1.drop(columns =  col)

data1 = data1.rename(columns= {'datetime' : 'datetime_local'})



chrt = plt.plot(data1.datetime_local,data1.power)
plt.show()





data = pd.merge(data2, data1, on = "datetime_local")




data_final = data.iloc[:,3:33]

###removing below features as most of them have nan values and are having same value in all cells


cols = ["precip_type","wind_chill", "heat_index", "qpf","snow", "pop","fctcode","precip_accumulation"]

data_final = data_final.drop(columns = cols)


data_final['datetime_local'] = pd.to_datetime(data_final['datetime_local'])


data_final['Hour'] = pd.DatetimeIndex(data_final['datetime_local']).hour

data_final = data_final[((data_final['Hour'] >= 6) & (data_final['Hour'] <=18))]


column = ['datetime_local']
          
data_final = data_final.drop(columns = column) 

data_final = data_final.reset_index()
data_final = data_final.drop(columns ='index')

power = data_final.power
power = power.reset_index()
power = power.drop(columns = 'index')
data_final = data_final.drop(columns ='power')
data_final = pd.concat([data_final, power], axis =1)


############################categorical features#########################

cat = ['sunrise', 'sunset','icon','summary','updated_at']

data_final = data_final.drop(columns = cat)


###converting Hour and Month into categorical feature

data_final.Hour = data_final.Hour.astype('category')
#data_final.Month = data_final.Month.astype('category')


######################################EDA#############################

data_final.info()

missing = data_final.isnull().sum()

data_final.icon.unique()
data_final.summary.nunique()
data_final.sunrise.nunique()

plt.hist(data_final['precip_probability'])


data_final.cloud_cover.value_counts()


############Outlier analysis##############

cols = ['power']

for i in cols:
    q75, q25 = np.percentile(data_final.loc[:,i], [75,25])
    iqr = q75 - q25
    
    minimum  = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    data_final.loc[data_final.loc[:,i] < minimum, i] = np.nan
    data_final.loc[data_final.loc[:,i] > maximum, i] = np.nan


missing = data_final.isnull().sum()

##########imputing values with mean#########



data_final.isnull().sum()

 for i in cols:
      data_final[i] = data_final[i].fillna(data_final[i].mean())



####################################normalization#######################################
      
      
     
cols = data_final.iloc[:,0:14].columns
         
 for p in cols:
     data_final[p] = (data_final[p] - data_final[p].min())/((data_final[p].max()) - (data_final[p].min()))


corr= data_final.corr()


##dropping some feature as it is correleatd to 3 variables

cols1 =['ozone',"wind_speed","apparent_temperature","precip_intensity","precip_probability","uv_index"]

data_final = data_final.drop(columns =cols1 )




data_final = pd.get_dummies(data_final)


#########################model #############################


x = data_final.drop(columns = 'power')
y = data_final.power


#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

x_train = x[:10000]
y_train = y[:10000]

x_test = x[10000:]
y_test = y[10000:]


##################################linear regression#########################

from statsmodels.api import OLS

model = OLS(y_train,x_train).fit()

model.summary()

predict_LR = model.predict(x_test)


def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true -y_pred) / (y_true)))
    return mape

error_LR = MAPE(y_test, predict_LR)




lrm_regressor = LinearRegression()
lrm_regressor.fit(x_train, y_train)
Y_predict_lrm =lrm_regressor.predict(x_test)


error = mean_squared_error(predict_LR, y_test)

######cross-validation#####


linreg = LinearRegression()

scores_LR = cross_val_score(linreg, x, y, cv=5, scoring='neg_mean_squared_error')




########################SVM###########################

regressor = SVR()
regressor.fit(x_train,y_train)
#5 Predicting a new result
y_pred = regressor.predict(x_test)


error_LR = MAPE(y_test, y_pred)


y_pred = y_pred.rename(columns= { 0 : 'powerPred'})


error = mean_squared_error(y_pred, y_test)



########cross validation##########

svmreg = SVR()

scores_SVM = cross_val_score(svmreg, x, y, cv=5, scoring='neg_mean_squared_error')



import seaborn as sns

ax1= sns.distplot(y_pred[1000:2000], color='b', label = 'predicted')
sns.distplot(y_test[1000:2000], color = 'r', ax = ax1, label = 'actual')





###############################test set##############################



test = pd.read_csv('weather_forecast.csv')


###droppping unnecessary columns 


TestCol = ['humidity','precip_type',"precip_intensity","precip_probability",'Unnamed: 0','plant_id','datetime_utc','apparent_temperature','wind_chill','heat_index','qpf','snow','pop','fctcode','uv_index','ozone', ]

test = test.drop(columns = TestCol)

Testcat = ['sunrise', 'sunset','icon','summary','updated_at']

test = test.drop(columns = Testcat)

test = test.drop(columns ='precip_accumulation')



test['datetime_local'] = pd.to_datetime(test['datetime_local'])


test['Hour'] = pd.DatetimeIndex(test['datetime_local']).hour

#test = test[(test['Hour'] >= 6) & (test['Hour'] <=18)]



test = test.reset_index()


test = test.drop(columns =  'index')


column = ['datetime_local']
          
test = test.drop(columns = column) 

missiing_test = test.isnull().sum()



test.Hour = test.Hour.astype('category')

test= pd.get_dummies(test)

cols= test.iloc[:,0:8].columns
     
 for p in cols:
     test[p] = (test[p] - test[p].min())/((test[p].max()) - (test[p].min()))



predict = pd.DataFrame(regressor.predict(test))

predict.rename(columns = {0: 'forecasted power'},inplace =True)
test_results = predict.to_csv('predict.csv')

predict.plot()





























