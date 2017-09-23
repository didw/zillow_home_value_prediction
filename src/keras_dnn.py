# imports 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

################
################
##    OLS     ##
################
################

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which I (Andy Harless) updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols
np.random.seed(17)
random.seed(17)

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv("../input/properties_2016.csv")
submission = pd.read_csv("../input/sample_submission.csv")
print(len(train),len(properties),len(submission))

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(df.median(), inplace=True)
    return df

def MAE(y, ypred):
    #logerror=log(Zestimate)-log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])

def baseline_model():
    model = Sequential()
    model.add(Dense(1, input_dim=55, kernel_initializer='glorot_normal', activation='sigmoid'))
    #model.add(Dense(1, kernel_initializer='glorot_normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

print(np.shape(train))
print(np.shape(y))
train = np.array(train)
y = np.array(y)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
train = scaler_x.fit_transform(train)
y = scaler_y.fit_transform(y.reshape(-1,1)).reshape(-1)


reg = LinearRegression()
print('fit LR...')
reg.fit(train, y)
print(MAE(scaler_y.inverse_transform(y), scaler_y.inverse_transform(reg.predict(train))))


reg = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=50, verbose=True)
#reg = LinearRegression()
#kfold = KFold(n_splits=10, random_state=17)
#results = cross_val_score(reg, train, y, cv=kfold)
#print("Results: %.7f (%.7f) MSE" % (results.mean(), results.std()))

print('fit keras...')
reg.fit(train, y)
print(MAE(scaler_y.inverse_transform(y), scaler_y.inverse_transform(reg.predict(train))))
train = [];  y = [] #memory

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']


########################
########################
##  Combine and Save  ##
########################
########################

print( "\npredicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    test_feed = np.array(get_features(test))
    test_feed = scaler_x.transform(test_feed)
    pred = reg.predict(test_feed)
    pred = scaler_y.inverse_transform(pred.reshape(-1,1)).reshape(-1)
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
print( submission.head() )


##### WRITE THE RESULTS
from datetime import datetime
print( "\nWriting results to disk ..." )
submission.to_csv('sub_keras_10{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFinished ...")
