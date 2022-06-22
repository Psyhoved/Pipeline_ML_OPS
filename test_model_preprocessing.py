import pandas as pd
from model_preprocessing import cut_num_split

train = pd.read_csv('data_sets/Train.csv')
target = pd.read_csv('data_sets/Target.csv')
test = pd.read_csv('data_sets/Test.csv')

print(80 * '*')
print('preparing data')
data = pd.concat([train, test])


def test_cut_num_split():
    df_cat, df_num = cut_num_split(data)

    assert len(df_cat.columns) == 4
    assert len(df_num.columns) == 12
