import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

X_test = pd.read_csv('interim/X_test.csv')
y_test = pd.read_csv('interim/y_test.csv')

model = joblib.load("models/ridge_model.pkl")

y_predict = model.predict(X_test)

print('Ошибка на тестовых данных')
print('MSE: %.1f' % mse(y_test, y_predict))
print('RMSE: %.1f' % mse(y_test, y_predict, squared=False))
print('R2 : %.4f' % r2_score(y_test, y_predict))
