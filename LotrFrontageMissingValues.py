import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

df=pd.concat([df_train,df_test],sort=False, ignore_index=True)

description = df.describe()
NanCount=df.isnull().sum()
ColumnName = df.columns.tolist()
NanColumn = []
for i in range(len(ColumnName)):
    if NanCount[i] > 0.3*len(df):
        NanColumn.append(ColumnName[i])
df.drop(NanColumn,axis=1, inplace= True)

correlation =df.corr()

Lot=['LotFrontage','LotArea','LotShape','LotConfig','MSSubClass','Neighborhood']
df=df[Lot]
Shape = df['LotShape'] == 'Reg'
df=df[Shape]
df=pd.get_dummies(df,prefix={'Neighborhood','LotConfig'}, columns = {'Neighborhood','LotConfig'})
EmptyLF= df['LotFrontage'].isnull()

Training = df[~EmptyLF]
Testing = df[EmptyLF]


data= Training.drop(['LotFrontage','LotShape'],axis=1)
target=Training.LotFrontage

X_train, X_val, y_train, y_val = train_test_split (data, target, random_state = 0)

model=RandomForestRegressor()
grid_params={'n_estimators':[50,70,100], 'max_depth':[3,5,7,10]}
RFReg = GridSearchCV(model,grid_params)
RFReg.fit(X_train, y_train)

print('Rand Forest Reg train',RFReg.score(X_train,y_train))
print('Rand Forest Reg test',RFReg.score(X_val,y_val))
print(RFReg.best_params_)


