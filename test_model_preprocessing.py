import pandas as pd


def test_cut_num_split():
    train = pd.read_csv('projects/SF_regression/data_sets/data_sets/Train.csv')
    test = pd.read_csv(r'projects/SF_regression/data_sets/data_sets/Test.csv')

    data = pd.concat([train, test])

    cat_columns = ['code', 'year', 'Country', 'id']
    num_columns = ['tourists', 'venue', 'rate', 'food', 'glass', 'metal', 'other',
                   'paper', 'plastic', 'leather', 'green_waste', 'waste_recycling']

    df_num = data[num_columns]
    df_cat = data[cat_columns]

    assert len(df_cat.columns) == 4
    assert len(df_num.columns) == 12

