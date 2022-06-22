import pandas as pd

def cut_num_split(data):
    cat_columns = ['code', 'year', 'Country', 'id']
    num_columns = ['tourists', 'venue', 'rate', 'food', 'glass', 'metal', 'other',
                   'paper', 'plastic', 'leather', 'green_waste', 'waste_recycling']
    df_num = data[num_columns]
    df_cat = data[cat_columns]

    return df_cat, df_num


train = pd.read_csv('https://drive.google.com/file/d/1-4uU9MGJMxpJuCLGJZKjYIHdA8daQXn_/view?usp=sharing')
target = pd.read_csv('https://drive.google.com/file/d/1-4YBjy125NchD6ityBMNQQH8eTYw_k-q/view?usp=sharing')
test = pd.read_csv('https://drive.google.com/file/d/1-4j2gKVZtrBtQahQfiw0h1Duoq72Ywwt/view?usp=sharing')

print(80 * '*')
print('preparing data')
data = pd.concat([train, test])


def test_cut_num_split():
    df_cat, df_num = cut_num_split(data)

    assert len(df_cat.columns) == 4
    assert len(df_num.columns) == 12
