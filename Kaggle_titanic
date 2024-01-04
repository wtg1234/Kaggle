import pandas as pd
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = df_train.drop(labels = ['Name','Ticket','Cabin'], axis = 1)
df_test = df_test.drop(labels = ['Name','Ticket','Cabin'], axis = 1)

df_test.Fare = df_test.Fare.fillna(7.7500)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df_train.Sex)
df_train.Sex = encoder.transform(df_train.Sex)
encoder.fit(df_test.Sex)
df_test.Sex = encoder.transform(df_test.Sex)

df_train.Age = df_train.Age.fillna(df_train.Age.median())
df_test.Age = df_test.Age.fillna(df_test.Age.median())

df_train = df_train.dropna(axis = 0)
df_test = df_test.dropna(axis = 0)

encoder.fit(df_train.Embarked)
df_train.Embarked = encoder.transform(df_train.Embarked)
encoder.fit(df_test.Embarked)
df_test.Embarked = encoder.transform(df_test.Embarked)

x_train = df_train.drop(labels = 'Survived',axis = 1)
y_train = df_train.Survived
x_test = df_test

import xgboost as xgb
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
params = {}
xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = 400,early_stopping_rounds = 100,
                      evals = [(dtrain,'train')])

y_pred_proba = xgb_model.predict(dtest)
y_pred = [1 if x>0.5 else 0 for x in y_pred_proba]

submission2 = pd.DataFrame({'PassengerID' : df_test.PassengerId, 'Survived' : y_pred})
submission2.to_csv('titanic2.csv',index = False)
submission2
