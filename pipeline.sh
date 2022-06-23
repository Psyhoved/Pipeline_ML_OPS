#/bin/bash
./data_creation.sh
python model_preprocessing.py
python model_preparation.py
python model_testing.py
python test_ model_preprocessing.py