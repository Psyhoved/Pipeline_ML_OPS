import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

X_train = pd.read_csv('interim/X_train.csv')
y_train = pd.read_csv('interim/y_train.csv')
X_test = pd.read_csv('interim/X_test.csv')
y_test = pd.read_csv('interim/y_test.csv')

alpha=0.2

model = Ridge(alpha=alpha, max_iter=10000)

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print('Ошибка на тестовых данных')
print('MSE: %.1f' % mse(y_test,y_predict))
print('RMSE: %.1f' % mse(y_test,y_predict,squared=False))
print('R2 : %.4f' %  r2_score(y_test,y_predict))
