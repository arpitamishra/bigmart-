import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import numpy as np
train = pd.read_csv("C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\big mart\\Train.csv")
test = pd.read_csv("C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\big mart\\Test.csv")
#print(train)
#print(test)
print(train.isnull().sum())
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)
X = train.drop(['Item_Outlet_Sales'], axis=1)
y = train.Item_Outlet_Sales
categorical_features_indices = np.where(X.dtypes != np.float)[0]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)
print(X_train)
print(y_train)
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features = categorical_features_indices)
result = pd.DataFrame()
result['Item_Identifier'] = test['Item_Identifier']
result['Outlet_Identifier'] = test['Outlet_Identifier']
result['Item_Outlet_Sales'] = model.predict(test)
result.to_csv("C:\\Users\\RAJIV MISHRA\\Desktop\\Arpita\\sampleproblemdataset\\big mart\\Submission.csv")



