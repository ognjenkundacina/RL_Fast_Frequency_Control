import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import time
import pickle
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


df = pd.read_csv('data_rl.csv')
df[['next_freq']] = df[['freq']].shift(-1)
df[['next_rocof']] = df[['rocof']].shift(-1)
#print(df.count())
df = df.drop(df[df.time == 1.00].index)
#print(df.count())
#print(df.head())

#print(df.loc[0, 'distur'])

y2 = df[['next_freq', 'next_rocof']] 
X2 = df[['distur', 'freq', 'rocof']]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.000001, random_state=28)

#RandomForestRegressor
clf = MultiOutputRegressor(RandomForestRegressor(random_state=1))
print('training regression started')
t1 = time.time()
clf.fit(X_train2, y_train2)
t2 = time.time()
print ('training regression finished in', t2-t1)
y_pred2 = clf.predict(X_test2)

pickle.dump(clf, open('regression.sav', 'wb'))

fig2 = plt.figure(figsize=(6,6))
plt.plot(y_test2.next_freq, y_test2.next_freq, c='k')
plt.scatter(y_test2.next_freq, y_pred2[:,0], c='g')
plt.xlabel('Observed')
plt.ylabel("Predicted")
plt.title("Observed vs predicted freq")
fig2.savefig('regression_1.png')

fig3 = plt.figure(figsize=(6,6))
plt.plot(y_test2.next_rocof, y_test2.next_rocof, c='k')
plt.scatter(y_test2.next_rocof, y_pred2[:,1], c='g')
plt.xlabel('Observed')
plt.ylabel("Predicted")
plt.title("Observed vs predicted rocof")
fig3.savefig('regression_2.png')

