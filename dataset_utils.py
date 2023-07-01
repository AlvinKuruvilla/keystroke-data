import os
import pandas as pd
import numpy as np


def read_compact_format():
    return pd.read_csv(
        os.path.join(os.getcwd(), "cleaned.csv"),
        dtype={
            "key": str,
            "press_time": np.float64,
            "release_time": np.float64,
            "platform_id": np.uint8,
            "session_id": np.uint8,
            "user_ids": np.uint8,
        },
    )


def get_index_of_last_non_matching_time_pair(df):
    last_index = None
    user_id = None
    for index, row in df.iterrows():
        if row["press_time"] != row["release_time"]:
            last_index = index
            user_id = row["user_ids"]
    return last_index, user_id


df = read_compact_format()
index, last_user_id = get_index_of_last_non_matching_time_pair(df)
print(index, last_user_id)
