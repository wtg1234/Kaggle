import pandas as pd
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = df_train.drop(labels=['Ticket','Cabin','Name'],axis = 1)
df_test = df_test.drop(labels = ['Ticket','Cabin','Name'],axis = 1)

df_train = pd.get_dummies(df_train, columns=['Sex'])
df_test = pd.get_dummies(df_test, columns=['Sex'])

df_train['Embarked'] = df_train['Embarked'].fillna(method = 'ffill')
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Embarked'] = df_test['Embarked'].fillna(method = 'ffill')
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Fare'] = df_test['Fare'].fillna(7.7500)

df_train = pd.get_dummies(df_train, columns = ['Embarked'])
df_test = pd.get_dummies(df_test, columns = ['Embarked'])

x_train = df_train.drop(labels = 'Survived',axis = 1)
y_train = df_train.Survived
x_test = df_test

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred1 = lr.predict(x_test)
print(lr.score(x_test,y_pred1))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_sc,y_train)
y_pred2 = lr.predict(x_test_sc)
print(lr.score(x_test_sc,y_pred2))

submission = pd.DataFrame({'PassengerID' : df_test.PassengerId, 'Survived' : y_pred2})
submission
submission.to_csv('titanic_2_sc.csv',index = False)

# import xgboost as xgb
# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)
# params = {}
# xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = 400,early_stopping_rounds = 100,
#                       evals = [(dtrain,'train')])
# y_pred_proba = xgb_model.predict(dtest)
# y_pred = [1 if x>0.5 else 0 for x in y_pred_proba]

# submission = pd.DataFrame({'PassengerID' : df_test.PassengerId, 'Survived' : y_pred})
# submission
# submission.to_csv('titanic_1.csv',index = False)
