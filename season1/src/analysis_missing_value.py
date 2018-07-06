# imports 
import numpy as np
import pandas as pd

##### READ IN RAW DATA
print( "\nReading data from disk ...")
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")

df_train = train.merge(prop, how='left', on='parcelid')

print(df_train['bedroomcnt'])
print(df_train['bedroomcnt']==0)
df_train.loc[df_train['bedroomcnt']==0] = np.NaN
df_train.iloc[:,2:] = df_train.iloc[:,2:].notnull().astype('int')

print(df_train.corr())