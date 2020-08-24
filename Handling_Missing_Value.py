# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:59:21 2020

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df=pd.concat([df_train,df_test],sort=False)


description = df.describe()
NanCount=df.isnull().sum()
ColumnName = df.columns.tolist()
NanColumn = []

for i in range(len(ColumnName)):
    if NanCount[i] > 0.3*len(df):
        NanColumn.append(ColumnName[i])
df.drop(NanColumn,axis=1, inplace= True)

ColumnName = df.columns.tolist()


#import missingno as msno
#msno.matrix(df)
#msno.heatmap(df_train)




## REPLACE OUTLIER BY YEAR BUILT
#print(df[['GarageArea','GarageType']].groupby('GarageType').agg(lambda x: x.mean()))
df['GarageYrBlt'].replace(2207,2007, inplace=True)

# FILLNA OF DETCHD GARAGE TYPE
# 1983 YEAR OF REMOD
# 2.0 MODE GARAGECARS
# 419 MEAN GARAGE AREA
# TA MODE GARAGE COND AND GARAGE QUAL

Detchd = df['GarageType'] == 'Detchd'
mean=df.loc[Detchd,'YearBuilt'].mean()
values = {'GarageYrBlt':1983,'GarageCars':2.0,'GarageArea':419,'GarageFinish':'Unf','GarageQual':'TA',
          'GarageCond':'TA'}
df.loc[Detchd, :] = df.loc[Detchd, :].fillna(value=values)

# FILNA IN NA GARAGE TYPE 
# 0 GARAGEYRBLT GARAGECARS GARAGEAREA
# NA OTHER CATEORGICAL FEATURES


values = {'GarageType': 'NA', 'GarageYrBlt':0,'GarageFinish':'NA',
          'GarageQual':'NA','GarageCond':'NA', 'GarageCars':0,'GarageArea':0}
df.fillna(value=values, inplace=True)


## MSZONING FILLNA ACORDING TO THE MODE OF EACH NEIGHBORHOOD
#print(df[['Neighborhood','MSZoning']].groupby('Neighborhood').agg(lambda x: x.mode()))


Mitchel= df['Neighborhood'] == 'Mitchel'
df.loc[Mitchel,'MSZoning']=df.loc[Mitchel,'MSZoning'].fillna('RL')


IDOTRR = df['Neighborhood'] == 'IDOTRR'
df.loc[IDOTRR,'MSZoning']=df.loc[IDOTRR,'MSZoning'].fillna('RM')

import missingno as msno
#msno.matrix(df)
#msno.heatmap(df)
#sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

Nanc= df.isnull().sum()



