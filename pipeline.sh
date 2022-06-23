#/bin/bash
./data_creation.sh
python model_preprocessing.py
python model_preparation.py
python model_testing.py
#pytest test_model_preprocessing.py