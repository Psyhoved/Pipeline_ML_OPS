#/bin/bash
kaggle competitions download -c sf-matml-2022-regression
mv sf-matml-2022-regression.zip ~/projects/SF_regression/data_sets/
unzip data_sets/sf-matml-2022-regression.zip -d data_sets

rm data_sets/sf-matml-2022-regression.zip
