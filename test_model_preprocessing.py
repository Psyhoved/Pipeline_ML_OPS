import pandas as pd


def test_cut_num_split():
    train = pd.read_csv('https://www.dropbox.com/s/o1jmfe26ri31k3l/Train.csv?dl=0')
    test = pd.read_csv(r'https://www.dropbox.com/s/qyajy9n51foaa69/Test.csv?dl=0')

    data = pd.concat([train, test])

    cat_columns = ['code', 'year', 'Country', 'id']
    num_columns = ['tourists', 'venue', 'rate', 'food', 'glass', 'metal', 'other',
                   'paper', 'plastic', 'leather', 'green_waste', 'waste_recycling']

    df_num = data[num_columns]
    df_cat = data[cat_columns]

    assert len(df_cat.columns) == 4
    assert len(df_num.columns) == 12

