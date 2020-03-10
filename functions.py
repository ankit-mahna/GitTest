#!/usr/bin/env python3

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Error metrics for Regression problems - MAPE and WMAPE


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def WMAPE(y_true, y_pred):
    return np.sum((np.abs((y_true - y_pred) / y_true) * 100 * y_true))/np.sum(y_true)


# Defining function for label encoding. Takes full data frame and list of object variables as input and
# returns a data frame with label encoded columns and others actual columns as well


def label_encoder(data, obj_list):
    encoded_lst = []
    le = LabelEncoder()
    for var in obj_list:
        encoded_lst.append(le.fit_transform(data[var]).tolist())
    le_df = pd.DataFrame(np.array(encoded_lst).T, columns = obj_list)
    data = data.drop(columns = obj_list)
    data = data.merge(le_df, left_index = True, right_index = True, how = 'inner')
    return data


# Function to call in case nltk is not downloading


def nltk_not_downloading():
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


