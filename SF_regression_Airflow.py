import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

import joblib
import datetime as dt
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

import warnings

warnings.simplefilter("ignore", UserWarning)

args = {'owner': 'airflow',
        'start_date': dt.datetime(2022, 6, 9),
        'retries': 1,
        'retry_delay': dt.timedelta(minutes=1),
        'depends_on_past': False,
        }


def create_dataset():
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


def train_model():
    X_train = pd.read_csv('interim/X_train.csv')
    y_train = pd.read_csv('interim/y_train.csv')

    alpha = 0.2

    model = Ridge(alpha=alpha, max_iter=10000)

    model.fit(X_train, y_train)

    joblib.dump(model, "models/ridge_model.pkl")


def make_prediction():
    X_test = pd.read_csv('interim/X_test.csv')
    y_test = pd.read_csv('interim/y_test.csv')

    model = joblib.load("models/ridge_model.pkl")

    y_predict = model.predict(X_test)

    print('Ошибка на тестовых данных')
    print('MSE: %.1f' % mse(y_test, y_predict))
    print('RMSE: %.1f' % mse(y_test, y_predict, squared=False))
    print('R2 : %.4f' % r2_score(y_test, y_predict))


with DAG(dag_id='test_airflow', default_args=args, shedule_interval=None) as dag:
    create_dataset = PythonOperator(task_id='create_dataset',
                                    python_callable=create_dataset,
                                    dag=dag)

    train_model = PythonOperator(task_id='train_model',
                                 python_callable=train_model,
                                 dag=dag)
    make_prediction = PythonOperator(task_id='make_prediction',
                                     python_callable=make_prediction,
                                     dag=dag)

    create_dataset >> train_model >> make_prediction
