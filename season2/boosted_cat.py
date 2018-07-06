import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

###########################################
# Data loading, we parse recordingdate
#############################################

def make_test():
    year = 2017
    # load data
    submission = pd.read_csv('../Data/sample_submission.csv')
    submission['parcelid'] = submission['ParcelId']
    print("submission data: ", submission.shape)

    tax_df = pd.read_csv('../Data/tax_history_2007to2017_v2.csv')
    print("tax data: ", tax_df.shape)

    home_attr = pd.read_csv('../Data/home_attributes_history/home_attributes_%d_v1.csv'%year)
    test_temp = pd.merge(submission,home_attr,how='left',on=['parcelid'])
    print("test_temp:", test_temp.shape)
    
    tax_temp = tax_df[tax_df['assessmentyear']==year].iloc[:,1:]
    test = pd.merge(test_temp,tax_temp,how='left',on='parcelid')
    test.drop(['parcelid'], inplace=True, axis=1)

    print("test:", test.shape)

    return test

train_df = pd.read_csv('../Data/merged/train.csv', parse_dates=['recordingdate'], low_memory=False)
#test_df = pd.read_csv('../input/sample_submission.csv', low_memory=False)
test_df = make_test()
#properties = pd.read_csv('../input/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']


# similar to the1owl
def add_date_features(df):
    df["transaction_year"] = df["recordingdate"].dt.year
    df["transaction_month"] = df["recordingdate"].dt.month
    df["transaction_day"] = df["recordingdate"].dt.day
    df["transaction_quarter"] = df["recordingdate"].dt.quarter
    df.drop(["recordingdate"], inplace=True, axis=1)
    return df

train_df = add_date_features(train_df)
#train_df = train_df.merge(properties, how='left', on='parcelid')
#test_df = test_df.merge(properties, how='left', on='parcelid')
print("Train: ", train_df.shape)
print("Test: ", test_df.shape)

############################################
# 0.a) Remove missing data fields
###########################################

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

#######################################
# 0.b) Remove data that is always the same
###########################################

# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

############################################
# 1.a) Define training features
#############################################
exclude_other = ['regionidcounty', 'transactionyear', 'recordingdate', 'parcelid',
       'legalrecordingid', 'createdate', 'documenttypeid',
       'concurrentloanamount', 'concurrentloancount', 'dataclasstypeid',
       'deedsloanamount', 'loanamount', 'loancount', 'saleprice',
       'partialinteresttransferpercent', 'partialinteresttransfertypeid',
       'inclusionruleidzestimate', 'derivedloanamount', 'derivedloancount'
       ]  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

####################################################
# 1.b) Define which of these training features are categorical
#################################################################

cat_feature_inds = []
cat_unique_thresh = 1300
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

# ['lotsiteappeals', 'architecturalstyletypeid', 'roofstructuretypeid', 'buildingqualitytypeid', 'propertylandusetypeid', 'regionidstate', 'decktypeid', 'heatingorsystemtypeid', 'typeconstructiontypeid', 'buildingconditiontypeid', 'watertypeid', 'fips', 'airconditioningtypeid', 'lotsizetopographytypeid', 'foundationtypeid', 'basementtypeid26', 'basementtypeid21', 'effectiveyearbuilt', 'yearbuilt', 'sewertypeid', 'regionidmsa', 'roofcovertypeid', 'storytypeid', 'yearremodelled', 'assessmentyear', 'latestdataatyear', 'taxratecodearea', 'edition', 'taxdelinquencyyear', 'transaction_year', 'transaction_month', 'transaction_day', 'transaction_quarter']
#######################################
# 1.c) Fill missing values
####################################

# some out of range int is a good choice
######################################
# 2.a) Training time!
#################
X_train = train_df[train_features]
y_train = train_df.saleprice
scaler = MinMaxScaler((0,10.))
y_train = scaler.fit_transform(y_train.values.reshape(-1,1)).reshape(-1)
print(X_train.shape, y_train.shape)

test_df['recordingdate'] = pd.Timestamp('2017-10-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)

# fill na with median
for c in X_train.columns:
	try:
		X_train[c].fillna(X_train[c].median(), inplace=True)
		X_test[c].fillna(X_test[c].median(), inplace=True)
	except TypeError:
		X_train[c].fillna(0, inplace=True)
		X_test[c].fillna(0, inplace=True)

# convert numpy.float64 to str
for i in cat_feature_inds:
	c = train_features[i]
	if type(X_train[c][0]) == np.float64 or type(X_test[c][0]) == np.float64:
		X_train.loc[:,c] = pd.Series(map(int,X_train.loc[:,c]))
		X_test.loc[:,c] = pd.Series(map(int,X_test.loc[:,c]))
	if type(X_train[c][0]) == np.str or type(X_test[c][0]) == np.str:
		X_train.loc[:,c] = pd.Series(map(str,X_train.loc[:,c]))
		X_test.loc[:,c] = pd.Series(map(str,X_test.loc[:,c]))
	if type(X_train[c][0])!=type(X_test[c][0]):
		print(c, type(X_train[c][0]), type(X_test[c][0]))
	assert(type(X_train[c][0])==type(X_test[c][0]))

for i in cat_feature_inds:
	c = train_features[i]
	if type(X_train[c][0])!=type(X_test[c][0]):
		print(c, type(X_train[c][0]), type(X_test[c][0]))
	assert(type(X_train[c][0])==type(X_test[c][0]))

submission_major = 1
"""
1: nens:5, itr: 200, lr: 0.02, dep: 5, llr: 2
"""
for nens in [1, ]:
	for itr in [100, ]:
		for lr in [0.002,]:
			for dep in [5,]:
				for llr in [2,]:
					num_ensembles = nens
					y_pred = 0.0
					for i in tqdm(range(num_ensembles)):
					    # TODO(you): Use CV, tune hyperparameters
					    model = CatBoostRegressor(
					        iterations=itr, learning_rate=lr,
					        depth=dep, l2_leaf_reg=llr,
					        loss_function='MAE',
					        eval_metric='MAE',
					        random_seed=i)
					    model.fit(
					        X_train, y_train,
					        cat_features=cat_feature_inds)
					    y_pred += np.array(map(lambda x:max(0,x)model.predict(X_test)))
					y_pred /= num_ensembles
					y_pred = scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
					###########################################
					# 3.) Create submission
					##############################

					submission = pd.DataFrame({
					    'ParcelId': test_df['parcelid'],
					})
					# https://www.kaggle.com/c/zillow-prize-1/discussion/33899, Oct,Nov,Dec
					test_dates = {
					    '201709': pd.Timestamp('2017-09-30'),
					    '201710': pd.Timestamp('2017-10-31'),
					    '201809': pd.Timestamp('2018-09-30'),
					    '201810': pd.Timestamp('2018-10-31')
					}
					for label, test_date in test_dates.items():
					    print("Predicting for: %s ... " % (label))
					    # TODO(you): predict for every `test_date`
					    submission[label] = 1.08*y_pred if '2018' in label else y_pred

					submission.to_csv(
					    'submission_%03d.csv' % (submission_major),
					    float_format='%.4f',
					    index=False)
					print("Done! Good luck with submission #%d :)" % submission_major)
					print("num_ens: %d, iter: %d, lr: %d, depth: %d, l2_leaf_reg: %d" % 
						(nens, itr, lr, dep, llr))
					submission_major += 1