import pandas as pd
import joblib
from sklearn.linear_model import Ridge

X_train = pd.read_csv('interim/X_train.csv')
y_train = pd.read_csv('interim/y_train.csv')

alpha = 0.2

model = Ridge(alpha=alpha, max_iter=10000)

model.fit(X_train, y_train)

joblib.dump(model, "models/ridge_model.pkl")
