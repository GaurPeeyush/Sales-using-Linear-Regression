import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

train = pd.read_csv("train.csv")

y = train['Sales']
X = train[['Price', 'AdvCost']]

clf = LinearRegression()

model = clf.fit(X, y)
joblib.dump(clf, 'reg_model.pkl')
