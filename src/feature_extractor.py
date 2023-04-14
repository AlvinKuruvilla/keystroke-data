import os

import numpy as np
import pandas as pd


def remove_outliers_for_dictionary_data(data):
    data.sort()

    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    IQR = q3 - q1
    upper = q3 + (1.5 * IQR)
    lower = q1 - (1.5 * IQR)
    return [
        element
        for element in data
        if element > lower and element < upper and element > 0
    ]


def get_new_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
        usecols=["key", "direction", "time", "platform_ids"],
    )
    df = df[["direction", "key", "time", "platform_ids"]]
    return df


def facebook_data(raw_df):
    df = raw_df.loc[(raw_df["platform_ids"] == 1)]
    return df.drop("platform_ids", axis=1)


def instagram_data(raw_df):
    df = raw_df.loc[(raw_df["platform_ids"] == 2)]
    return df.drop("platform_ids", axis=1)


def twitter_data(raw_df):
    df = raw_df.loc[(raw_df["platform_ids"] == 3)]
    return df.drop("platform_ids", axis=1)


if __name__ == "__main__":
    raw_df = get_new_format()
    print(raw_df["direction"].unique().tolist())
