import json
import os

import numpy as np
import pandas as pd

from algo.algorithms import kit_features


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
    facebook_df = facebook_data(raw_df)
    instagram_df = instagram_data(raw_df)
    twitter_df = twitter_data(raw_df)

    # processed_KHT_data = {}
    # for key in list(data.keys()):
    #     res = remove_outliers_for_dictionary_data(data[key])
    #     processed_KHT_data[key] = res
    # # # TODO: Double check this doesn't have unexpected side-effects when removing keys with empty value lists
    # res = {k: v for k, v in processed_KHT_data.items() if v}
    for i in range(1, 5):
        data = kit_features(instagram_df, i)
        with open(
            os.path.join(os.getcwd(), "features", "raw_Instagram_KIT_" + i + ".json"),
            "w",
        ) as f:
            json.dump(data, f)
