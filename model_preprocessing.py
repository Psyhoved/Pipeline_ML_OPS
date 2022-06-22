import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter("ignore", UserWarning)

print('reading data')
train = pd.read_csv('data_sets/Train.csv')
target = pd.read_csv('data_sets/Target.csv')
test = pd.read_csv('data_sets/Test.csv')

print(80 * '*')
print('preparing data')
data = pd.concat([train, test])

cat_columns = ['code', 'year', 'Country', 'id']
num_columns = ['tourists', 'venue', 'rate', 'food', 'glass', 'metal', 'other',
               'paper', 'plastic', 'leather', 'green_waste', 'waste_recycling']
df_num = data[num_columns]
df_cat = data[cat_columns]

df_cat['year'] = df_cat['year'].astype('str')
# one hot encoding
df_cat = pd.get_dummies(df_cat)

# Добавляем везде индекс, для дальнейшего merge 
df_cat['idx'] = data['Unnamed: 0']
df_num['idx'] = data['Unnamed: 0']

prepare_data = pd.merge(df_num, df_cat)
len_test = len(test)

train_prep = prepare_data[:-len_test]
test_prep = prepare_data[-len_test:]

X, y = train_prep.values, target['polution'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(80 * '*')
print('saving train-test splited data in interim folder')

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X_train.to_csv('interim/X_train.csv')
X_test.to_csv('interim/X_test.csv')
y_train.to_csv('interim/y_train.csv')
y_test.to_csv('interim/y_test.csv')
