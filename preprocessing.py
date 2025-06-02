from numpy.typing import ArrayLike
import pandas as pd
import os
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(show_data=False)-> tuple[DataFrame, Series]:
    datapath = os.path.join(os.path.dirname(__file__), "data", "weather_classification_data.csv")
    if not os.path.isfile(datapath):
        raise FileNotFoundError(f'can\'t find file with data, {datapath} does not exist')
    
    data = pd.read_csv(datapath)
    if show_data:
        print(f'Raw data: {data.head()}')

    features: DataFrame = data.drop(data.columns[-1], axis=1)
    target: Series[str] = data[data.columns[-1]]
    return features, target

def encode_data(features: DataFrame, target: Series, show_encoded_data=False) -> tuple[DataFrame, ArrayLike, LabelEncoder]:
    features_encoded = pd.get_dummies(features)
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    if show_encoded_data:
        print(f'encoded features: {features_encoded.head()}')
        print('\n\n')
        print(f'encoded targets: {target_encoded}')

    return features_encoded, target_encoded, label_encoder

def split(features_encoded: DataFrame, target_encoded: Series | ArrayLike):
    return train_test_split(features_encoded, target_encoded, test_size=0.2, shuffle=True, random_state=10)
