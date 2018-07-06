import pandas as pd

# load data
saleprice = pd.read_csv('saleprice/saleprice_2007-01-01to2017-06-30_v3.csv')
sale_train = saleprice[saleprice['saleprice']>0]
sale_test = saleprice[not saleprice['saleprice']>0]
print("train data: ", sale_train.shape)
print("test data: ", sale_test.shape)

tax = pd.read_csv('tax_history_2007to2017_v2.csv')
print("tax data: ", tax.shape)

home_attribute = {}
for year in range(2007,2018):
    home_attribute[year] = pd.read_csv('home_attributes_history/home_attributes_%d_v1.csv'%year)
    print("home_attribute[%d]:"%year, home_attribute[year].shape)

for year in range(2007,2018):
    sale_temp = sale_train[sale_train['transactionyear']==year]
    print("[%d]sale_temp:"%year, sale_temp.shape)
    train_temp = pd.merge(sale_temp,home_attribute[year],how='inner',on=['parcelid'])
    print("[%d]train_temp:"%year, train_temp.shape)
    train = pd.merge(train_temp,tax,how='inner',on='parcelid')
    print("[%d]train:"%year, train.shape)

    sale_temp = sale_test[sale_test['transactionyear']==year]
    test_temp = pd.merge(sale_temp,home_attribute[year],how='inner',on=['parcelid'])
    test = pd.merge(test_temp,tax,how='inner',on='parcelid')
    print("[%d]test"%year, test.shape)

    train.to_csv('merged/train_%d.csv'%year, index=False)
    test.to_csv('merged/test_%d.csv'%year, index=False)
